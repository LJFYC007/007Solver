export type Player = "hero" | "villain";

export interface NodeState {
    board: string[];
    pot: number;
    stacks: {
        hero: number;
        villain: number;
    };
    street: "flop" | "turn" | "river";
    rangeCombos: Record<Player, number>;
}

export interface HandStrategy {
    cards: string[];
    inputRangeWeight: number;
    marginalReachMass: number;
    nodeStrategyEv: number | null;
    ownReachWeight: number;
    strategy: number[];
    /** EV of each action followed by the node strategy; empty exactly when nodeStrategyEv is null. */
    actionEvs: number[];
}

export interface DecisionAction {
    amountTo: number;
    chipsCommitted: number;
    isAllIn: boolean;
    kind: "fold" | "check" | "call" | "bet" | "raise";
    nextNodeId: number;
}

export interface DecisionNode {
    actions: DecisionAction[];
    actor: Player;
    hands: HandStrategy[];
    nodeId: number;
    kind: "decision";
    /** Each player's node strategy EV averaged over joint reach; null without joint reach. */
    rangeEvs: Record<Player, number | null>;
    state: NodeState;
}

export interface ChanceNode {
    nodeId: number;
    kind: "chance";
    outcomes: {
        card: string;
        nextNodeId: number;
    }[];
    /** Forced runouts only: the rake their showdowns pay. */
    rake?: number;
    state: NodeState;
}

export interface TerminalNode {
    nodeId: number;
    kind: "terminal";
    result:
        | {
              foldedBy: Player;
              reason: "fold";
          }
        | {
              reason: "showdown";
          };
    rake: number;
    state: NodeState;
}

export type SolverNode = ChanceNode | DecisionNode | TerminalNode;

export interface MemoryEstimate {
    logicalNodes: number;
    topologyNodes: number;
    traversalNodes: number;
    strategyEntries: number;
    peakBytes: number;
    workers: number;
}

export interface SolveProgress {
    phase: "training" | "checking" | "finalizing" | "complete";
    elapsedSeconds: number;
    estimatedRemainingSeconds: number | null;
    accuracyPercent: number | null;
    targetAccuracyPercent: number;
}

export type SolverStatus =
    | { state: "idle" }
    // Web sessions wait here while another solve holds the GPU.
    | { state: "queued" }
    | {
          state: "starting";
      }
    | {
          state: "buildingTree";
          totalIterations: number;
      }
    | (SolveProgress & {
          completedIterations: number;
          state: "solving";
          totalIterations: number;
          estimate?: MemoryEstimate | null;
      })
    | (SolveProgress & {
          iterations: number;
          nodeCount: number;
          rootNodeId: number;
          state: "ready";
          stopReason: "accuracy" | "iterationLimit";
          estimate?: MemoryEstimate | null;
      })
    | {
          message: string;
          state: "failed";
      };

/** The waiting player's node strategy EVs, a separate query because they take another subtree evaluation. */
export interface OpponentEvReport {
    nodeId: number;
    player: Player;
    hands: { cards: string[]; marginalReachMass: number; nodeStrategyEv: number | null }[];
}

export interface EquityReport {
    nodeId: number;
    players: Record<
        Player,
        {
            equity: number | null;
            hands: { cards: string[]; ownReachWeight: number; equity: number | null }[];
        }
    >;
}
