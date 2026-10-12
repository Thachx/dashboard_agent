import { randomUUID } from "node:crypto";
import { mkdir, readdir, readFile, stat, unlink, writeFile } from "node:fs/promises";
import path from "node:path";

export interface DashboardPreviewStore {
  put(image: Buffer): Promise<string>;
  get(id: string): Promise<Buffer | undefined>;
}

/**
 * Disk-backed PNG preview store. Previews survive service restarts so that
 * markdown images inside pasted chat history keep rendering until they expire.
 */
export function createDashboardPreviewStore(options: {
  directory: string;
  ttlMs: number;
}): DashboardPreviewStore {
  const directory = path.resolve(options.directory);
  let directoryReady = false;
  let lastSweepAt = 0;

  async function sweepExpired(now: number): Promise<void> {
    if (now - lastSweepAt < 3_600_000) return;
    lastSweepAt = now;
    let names: string[];
    try {
      names = await readdir(directory);
    } catch {
      return;
    }
    for (const name of names) {
      if (!name.endsWith(".png")) continue;
      const file = path.join(directory, name);
      try {
        const info = await stat(file);
        if (now - info.mtimeMs > options.ttlMs) {
          await unlink(file).catch(() => {});
        }
      } catch {
        continue;
      }
    }
  }

  async function ensureDirectory(): Promise<void> {
    if (!directoryReady) {
      await mkdir(directory, { recursive: true });
      directoryReady = true;
    }
  }

  function filePath(id: string): string {
    if (!/^[a-f0-9-]{16,64}$/i.test(id)) {
      throw new Error("Invalid dashboard preview id");
    }
    return path.join(directory, `${id}.png`);
  }

  async function isFresh(id: string): Promise<boolean> {
    try {
      const stat = await readFileStat(filePath(id));
      return Date.now() - stat.mtimeMs <= options.ttlMs;
    } catch {
      return false;
    }
  }

  return {
    async put(image) {
      await ensureDirectory();
      const now = Date.now();
      await sweepExpired(now);
      const id = randomUUID();
      await writeFile(filePath(id), image);
      return id;
    },

    async get(id) {
      try {
        const file = filePath(id);
        if (!(await isFresh(id))) {
          await unlink(file).catch(() => {});
          return undefined;
        }
        return await readFile(file);
      } catch {
        return undefined;
      }
    },
  };
}

function readFileStat(file: string): Promise<{ mtimeMs: number }> {
  return stat(file).then((s) => ({ mtimeMs: s.mtimeMs }));
}
