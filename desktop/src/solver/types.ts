export type Player = "hero" | "villain";

export interface NodeState {
    board: string[];
    pot: number;
    stacks: {
        hero: number;
        villain: number;
    };
    street: "flop" | "turn" | "river";
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

export type SolverStatus =
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
      }
    | {
          iterations: number;
          nodeCount: number;
          rootNodeId: number;
          state: "ready";
      }
    | {
          message: string;
          state: "failed";
      };
