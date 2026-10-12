import { webcrypto } from "node:crypto";

// Node 18 does not expose a global `crypto` (added in Node 19).  The dashboard
// contract uses the global `crypto` so it can be bundled into the browser UI and
// run in the Node server alike; only older test workers need this polyfill.
if (typeof globalThis.crypto === "undefined") {
  Object.defineProperty(globalThis, "crypto", {
    value: webcrypto,
    configurable: true,
  });
}
