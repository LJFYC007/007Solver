use postflop_solver::*;
use serde_json::{json, Value};
use std::{fs, path::Path};

const SCALE: f32 = 500.0;
const MAX_REFERENCE_ITERATIONS: u32 = 100_000;
const REVISION: &str = "9d1509fe5077d019825f833eed04b16d342dfda1";

// Apply the configured cap through the independent engine's public tree editing API.
// Opening bets are excluded; reset the raise count at chance boundaries.
fn cap_raises(tree: &mut ActionTree, raises: usize, maximum: usize, facing_bet: bool) {
    if tree.is_terminal_node() {
        return;
    }
    let raises = if tree.is_chance_node() { 0 } else { raises };
    let facing_bet = !tree.is_chance_node() && facing_bet;
    let actions = tree.available_actions().to_vec();
    for action in actions {
        let is_raise = matches!(action, Action::Raise(_))
            || (facing_bet && matches!(action, Action::AllIn(_)));
        if raises >= maximum && is_raise {
            tree.remove_action(action).unwrap();
            continue;
        }
        tree.play(action).unwrap();
        cap_raises(
            tree,
            raises + usize::from(is_raise),
            maximum,
            matches!(action, Action::Bet(_) | Action::Raise(_) | Action::AllIn(_)),
        );
        tree.undo().unwrap();
    }
}

fn game_for(scenario: &Value, scale: f32) -> PostFlopGame {
    assert_eq!(scenario["heroActsFirst"], false);
    assert_eq!(scenario["heroStack"], scenario["villainStack"]);
    let range = |position: &str| {
        scenario["ranges"][position]
            .as_object()
            .unwrap()
            .iter()
            .map(|(hand, weight)| format!("{hand}:{weight}"))
            .collect::<Vec<_>>()
            .join(",")
            .parse()
            .unwrap()
    };
    let cards = CardConfig {
        range: [
            range(scenario["villainPosition"].as_str().unwrap()),
            range(scenario["heroPosition"].as_str().unwrap()),
        ],
        flop: flop_from_str(&scenario["board"].as_str().unwrap().replace(' ', "")).unwrap(),
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };
    let percentages = |values: &Value| {
        values
            .as_array()
            .unwrap()
            .iter()
            .map(|percent| format!("{}%", percent.as_f64().unwrap()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let sizes = |street: &str| {
        let bets = percentages(&scenario["bettingTree"][street]["bet"]);
        let raises = percentages(&scenario["bettingTree"][street]["raise"]);
        let sizes = BetSizeOptions::try_from((bets.as_str(), raises.as_str())).unwrap();
        [sizes.clone(), sizes]
    };
    let tree = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: (scenario["initialPot"].as_f64().unwrap() * scale as f64).round() as i32,
        effective_stack: (scenario["heroStack"].as_f64().unwrap() * scale as f64).round() as i32,
        flop_bet_sizes: sizes("flop"),
        turn_bet_sizes: sizes("turn"),
        river_bet_sizes: sizes("river"),
        force_allin_threshold: scenario["bettingTree"]["allInSpr"].as_f64().unwrap(),
        // Defaults disable rake, additional all-ins and merging of nearby sizes.
        ..Default::default()
    };
    let mut tree = ActionTree::new(tree).unwrap();
    cap_raises(
        &mut tree,
        0,
        scenario["bettingTree"]["maxRaises"].as_u64().unwrap() as usize,
        false,
    );
    let mut game = PostFlopGame::with_config(cards, tree).unwrap();
    game.allocate_memory(false);
    game
}

fn metrics(game: &PostFlopGame, scale: f32) -> Value {
    let ev = compute_mes_ev(game);
    json!({
        "heroBestResponseEv": ev[1] / scale,
        "villainBestResponseEv": ev[0] / scale,
        "exploitability": compute_exploitability(game) / scale,
    })
}

fn hand_name((first, second): (Card, Card)) -> String {
    format!(
        "{} {}",
        card_to_string(first).unwrap(),
        card_to_string(second).unwrap()
    )
}

fn fixed_policy(game: &mut PostFlopGame) -> Value {
    let mut policy = Vec::new();
    for depth in 0..2 {
        let hands = game.private_cards(game.current_player());
        let mut check = Vec::new();
        for &(first, second) in hands {
            let ranks = [first / 4, second / 4];
            check.push(if depth == 0 {
                if ranks.contains(&12) {
                    0.25
                } else {
                    1.0
                } // A5s / KK
            } else if ranks.contains(&11) {
                0.0
            } else {
                1.0
            }); // AKs / QQ
        }
        let entries: Vec<_> = hands
            .iter()
            .zip(&check)
            .map(|(&hand, &p)| json!({"cards": hand_name(hand), "strategy": [p, 1.0 - p]}))
            .collect();
        let strategy: Vec<_> = check
            .iter()
            .copied()
            .chain(check.iter().map(|p| 1.0 - p))
            .collect();
        game.lock_current_strategy(&strategy);
        policy.push(json!({"path": vec![0; depth], "hands": entries}));
        if depth == 0 {
            game.play(0);
        }
    }
    game.back_to_root();
    json!(policy)
}

fn query_reference(game: &mut PostFlopGame, scenario: &Value, path: Value) -> Value {
    game.back_to_root();
    game.cache_normalized_weights();
    let initial_weights = [game.weights(0).to_vec(), game.weights(1).to_vec()];
    let joint_weight: f32 = game.normalized_weights(0).iter().sum();
    let mut chance_factor = 1.0;
    for step in path.as_array().unwrap() {
        if let Some(card) = step.as_str() {
            chance_factor *= (48 - game.current_board().len()) as f32;
            game.play(card_from_str(card).unwrap() as usize);
        } else {
            game.play(step.as_u64().unwrap() as usize);
        }
    }
    game.cache_normalized_weights();
    let player = game.current_player();
    let evs = game.expected_values(player);
    let strategy = game.strategy();
    let cards = game.private_cards(player);
    let board = game.current_board();
    let hands: Vec<_> = cards
        .iter()
        .enumerate()
        .filter(|(_, (a, b))| !board.contains(a) && !board.contains(b))
        .map(|(i, &hand)| {
            let mass = game.normalized_weights(player)[i] / (joint_weight * chance_factor);
            let probabilities: Vec<_> = if mass > 0.0 {
                (0..game.available_actions().len())
                    .map(|a| strategy[a * cards.len() + i])
                    .collect()
            } else {
                vec![]
            };
            json!({
                "cards": hand_name(hand),
                "inputRangeWeight": initial_weights[player][i],
                "ownReachWeight": game.weights(player)[i],
                "marginalReachMass": mass,
                "nodeStrategyEv": if mass > 0.0 { Some(evs[i] / SCALE) } else { None },
                "strategy": probabilities,
            })
        })
        .collect();
    let bets = game.total_bet_amount();
    json!({
        "path": path,
        // The external solver sorts the flop; the application preserves the input order.
        "board": scenario["board"].as_str().unwrap().split_whitespace().map(str::to_owned)
            .chain(board.iter().skip(3).map(|&c| card_to_string(c).unwrap())).collect::<Vec<_>>().join(" "),
        "actor": 1 - player,
        "pot": scenario["initialPot"].as_f64().unwrap() + (bets[0] + bets[1]) as f64 / SCALE as f64,
        "stacks": [scenario["heroStack"].as_f64().unwrap() - bets[1] as f64 / SCALE as f64,
                   scenario["villainStack"].as_f64().unwrap() - bets[0] as f64 / SCALE as f64],
        "hands": hands,
    })
}

fn main() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures");
    let mut output = json!({"source": {
        "repository": "https://github.com/b-inary/postflop-solver",
        "revision": REVISION, "chipScale": SCALE,
    }});
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--wide"] {
        let scenario: Value =
            serde_json::from_str(&fs::read_to_string(fixtures.join("utg-bb-wide.json")).unwrap())
                .unwrap();
        // Native integer rounding at scale 10 matches the application's tenths of a chip.
        // Disable threshold replacement: the upstream engine rounds its SPR threshold.
        let scale = 10.0;
        output["source"]["chipScale"] = json!(scale);
        assert_eq!(scenario["initialPot"], 26.5);
        assert_eq!(scenario["heroStack"], 87.0);
        assert_eq!(scenario["bettingTree"]["allInSpr"], 0.0);
        for street in ["flop", "turn", "river"] {
            assert_eq!(scenario["bettingTree"][street]["bet"], json!([33, 125]));
            assert_eq!(scenario["bettingTree"][street]["raise"], json!([50]));
        }
        let game = game_for(&scenario, scale);
        let uniform = metrics(&game, scale);
        for key in [
            "heroBestResponseEv",
            "villainBestResponseEv",
            "exploitability",
        ] {
            assert!(uniform[key].as_f64().unwrap().is_finite());
        }
        assert!(uniform["exploitability"].as_f64().unwrap() >= 0.0);
        // A fixed-work benchmark needs an exact prescribed-policy reference, not a solved equilibrium.
        output["utg-bb-wide"] = json!({"scenario": scenario, "uniform": uniform});
        fs::write(
            fixtures.join("benchmark-reference.json"),
            serde_json::to_string_pretty(&output).unwrap() + "\n",
        )
        .unwrap();
        return;
    }
    assert!(args.is_empty(), "Usage: solver-test-oracle [--wide]");
    for name in ["weighted-flop", "raise-flop"] {
        let scenario: Value = serde_json::from_str(
            &fs::read_to_string(fixtures.join(format!("{name}.json"))).unwrap(),
        )
        .unwrap();
        let mut game = game_for(&scenario, SCALE);
        let uniform = metrics(&game, SCALE);
        let exploitability = solve(&mut game, MAX_REFERENCE_ITERATIONS, SCALE * 1e-5, false);
        assert!(
            exploitability / SCALE <= 1e-5,
            "{name} reference did not converge: {}",
            exploitability / SCALE
        );
        let solved = metrics(&game, SCALE);
        let mut game = game_for(&scenario, SCALE);
        let policy = if name == "weighted-flop" {
            fixed_policy(&mut game)
        } else {
            json!([])
        };
        // No solve: finalize evaluates the prescribed policy, with uniform play elsewhere.
        finalize(&mut game);
        let paths = if name == "weighted-flop" {
            json!([
                [],
                [0],
                [0, 1],
                [0, 0, "As"],
                [0, 0, "Qc", 0],
                [0, 0, "Qc", 0, 0, "Th"]
            ])
        } else {
            json!([[1, 1, "Qc"], [1, 1, "Qc", 1]])
        };
        let queries: Vec<_> = paths
            .as_array()
            .unwrap()
            .iter()
            .map(|path| query_reference(&mut game, &scenario, path.clone()))
            .collect();
        output[name] =
            json!({"scenario": scenario, "solved": solved, "policy": policy, "queries": queries});
        if name == "weighted-flop" {
            output[name]["uniform"] = uniform;
        }
    }
    fs::write(
        fixtures.join("correctness-reference.json"),
        serde_json::to_string_pretty(&output).unwrap() + "\n",
    )
    .unwrap();
}
