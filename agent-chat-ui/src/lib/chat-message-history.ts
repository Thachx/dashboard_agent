import {
  BaseListChatMessageHistory,
} from "@langchain/core/chat_history";
import {
  type BaseMessage,
} from "@langchain/core/messages";

// Minimal in-memory chat history compatible with older ChatMessageHistory usage.
export class ChatMessageHistory extends BaseListChatMessageHistory {
  lc_namespace = ["langchain", "stores", "message", "chat_message_history"];
  private messages: BaseMessage[];

  constructor(messages: BaseMessage[] = []) {
    super();
    this.messages = messages;
  }

  async getMessages(): Promise<BaseMessage[]> {
    return this.messages;
  }

  async addMessage(message: BaseMessage): Promise<void> {
    this.messages.push(message);
  }

  async clear(): Promise<void> {
    this.messages = [];
  }
}
