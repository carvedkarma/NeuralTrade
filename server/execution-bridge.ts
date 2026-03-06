import { broadcast } from "./ws";

interface BybitPosition {
  symbol: string;
  side: string;
  size: string;
  entryPrice: string;
  markPrice: string;
  unrealisedPnl: string;
  leverage: string;
  positionIdx: number;
  takeProfit: string;
  stopLoss: string;
  trailingStop: string;
  liqPrice: string;
  createdTime: string;
  updatedTime: string;
}

interface BybitBalance {
  equity: string;
  walletBalance: string;
  availableToWithdraw: string;
  unrealisedPnl: string;
  totalMarginBalance: string;
}

interface ExecutionState {
  positions: BybitPosition[];
  balance: BybitBalance | null;
  timestamp: number;
}

class ExecutionBridge {
  private lastPushTime: number = 0;
  private cachedState: ExecutionState = {
    positions: [],
    balance: null,
    timestamp: 0,
  };

  pushState(state: ExecutionState): void {
    this.cachedState = state;
    this.lastPushTime = Date.now();

    broadcast("EXECUTION_STATE", {
      positions: state.positions,
      balance: state.balance,
      timestamp: state.timestamp,
    });
  }

  getState(): ExecutionState {
    return this.cachedState;
  }

  getPositions(): BybitPosition[] {
    return this.cachedState.positions;
  }

  getBalance(): BybitBalance | null {
    return this.cachedState.balance;
  }

  isConnected(): boolean {
    if (this.lastPushTime === 0) return false;
    return (Date.now() - this.lastPushTime) < 30_000;
  }

  getLastPushTime(): number {
    return this.lastPushTime;
  }

  getStatus(): {
    connected: boolean;
    lastPushTime: number;
    lastPushAgo: number | null;
    positionCount: number;
    hasBalance: boolean;
  } {
    const now = Date.now();
    return {
      connected: this.isConnected(),
      lastPushTime: this.lastPushTime,
      lastPushAgo: this.lastPushTime > 0 ? now - this.lastPushTime : null,
      positionCount: this.cachedState.positions.length,
      hasBalance: this.cachedState.balance !== null,
    };
  }
}

export const executionBridge = new ExecutionBridge();
