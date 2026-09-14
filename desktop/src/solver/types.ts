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
    state: NodeState;
}

export interface ChanceNode {
    nodeId: number;
    kind: "chance";
    outcomes: {
        card: string;
        nextNodeId: number;
    }[];
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

export type SolverStatus =
    | { state: "idle" }
    | {
          state: "starting";
      }
    | {
          state: "buildingTree";
          totalIterations: number;
      }
    | {
          completedIterations: number;
          state: "solving";
          totalIterations: number;
          estimate?: MemoryEstimate | null;
      }
    | {
          iterations: number;
          nodeCount: number;
          rootNodeId: number;
          state: "ready";
          estimate?: MemoryEstimate | null;
      }
    | {
          message: string;
          state: "failed";
      };

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
