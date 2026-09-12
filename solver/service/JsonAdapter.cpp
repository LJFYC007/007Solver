#include "service/JsonAdapter.h"
#include "core/Chips.h"
#include "core/PokerTypes.h"
#include "game/BettingAction.h"
#include <limits>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>

namespace solver::service
{
namespace
{
using Json = nlohmann::json;

double ToChipUnits(core::Chips chips)
{
    return static_cast<double>(chips.Raw()) / static_cast<double>(core::Chips::kUnitsPerChip);
}

std::string PlayerCode(core::PlayerId player)
{
    return player == core::PlayerId::Player0() ? "hero" : "villain";
}

std::string StreetCode(core::Street street)
{
    switch (street)
    {
    case core::Street::Flop:
        return "flop";
    case core::Street::Turn:
        return "turn";
    case core::Street::River:
        return "river";
    }
    throw std::logic_error("Unknown street");
}

std::string ActionCode(const analysis::ActionReport& action)
{
    switch (action.kind)
    {
    case game::BettingActionKind::Fold:
        return "fold";
    case game::BettingActionKind::Check:
        return "check";
    case game::BettingActionKind::Call:
        return "call";
    case game::BettingActionKind::Bet:
        return "bet";
    case game::BettingActionKind::Raise:
        return "raise";
    }
    throw std::logic_error("Unknown betting action kind");
}

Json FormatBoard(const core::Board& board)
{
    Json cards = Json::array();
    for (int position = 0; position < board.CardCount(); ++position)
        cards.push_back(core::FormatCard(board.CardAt(position)));
    return cards;
}

Json FormatHoleCards(core::HoleCards hand)
{
    const std::array<core::Card, 2> cards = hand.Cards();
    return Json::array({core::FormatCard(cards[0]), core::FormatCard(cards[1])});
}

Json BuildEquityJson(const analysis::EquityReport& report)
{
    Json players = Json::object();
    for (std::size_t player = 0; player < 2; ++player)
    {
        const auto& data = report.players[player];
        Json hands = Json::array();
        for (const auto& hand : data.hands)
            hands.push_back(
                {{"cards", FormatHoleCards(hand.cards)},
                 {"ownReachWeight", hand.ownReachWeight},
                 {"equity", hand.equity ? Json(*hand.equity) : Json(nullptr)}}
            );
        players[player == 0 ? "hero" : "villain"] = {
            {"equity", data.equity ? Json(*data.equity) : Json(nullptr)}, {"hands", std::move(hands)}
        };
    }
    return {{"nodeId", report.nodeId.Value()}, {"players", std::move(players)}};
}

Json BuildNodeJson(const analysis::NodeReport& node)
{
    Json jsonNode = {
        {"nodeId", node.nodeId.Value()},
        {"state",
         {
             {"street", StreetCode(node.state.street)},
             {"board", FormatBoard(node.state.board)},
             {"pot", ToChipUnits(node.state.pot)},
             {"rangeCombos", {{"hero", node.state.rangeCombos[0]}, {"villain", node.state.rangeCombos[1]}}},
             {"stacks",
              {
                  {"hero", ToChipUnits(node.state.stacks[0])},
                  {"villain", ToChipUnits(node.state.stacks[1])},
              }},
         }},
    };

    if (node.kind == game::NodeKind::Terminal)
    {
        jsonNode["kind"] = "terminal";
        if (node.terminal->kind == game::TerminalKind::Fold)
        {
            jsonNode["result"] = {
                {"reason", "fold"},
                {"foldedBy", PlayerCode(*node.terminal->foldedPlayer)},
            };
        }
        else
        {
            jsonNode["result"] = {{"reason", "showdown"}};
        }
        return jsonNode;
    }

    if (node.kind == game::NodeKind::Chance)
    {
        jsonNode["kind"] = "chance";
        jsonNode["outcomes"] = Json::array();
        for (const analysis::ChanceOutcomeReport& outcome : node.outcomes)
        {
            jsonNode["outcomes"].push_back({
                {"card", core::FormatCard(outcome.card)},
                {"nextNodeId", outcome.nextNodeId.Value()},
            });
        }
        return jsonNode;
    }

    jsonNode["kind"] = "decision";
    jsonNode["actor"] = PlayerCode(*node.actor);
    jsonNode["hands"] = Json::array();
    for (const analysis::HandReport& hand : node.hands)
    {
        jsonNode["hands"].push_back({
            {"cards", FormatHoleCards(hand.cards)},
            {"inputRangeWeight", hand.inputRangeWeight},
            {"ownReachWeight", hand.ownReachWeight},
            {"marginalReachMass", hand.marginalReachMass},
            {"nodeStrategyEv", hand.nodeStrategyEv.has_value() ? Json(*hand.nodeStrategyEv) : Json(nullptr)},
            {"strategy", hand.strategy},
        });
    }

    jsonNode["actions"] = Json::array();
    for (const analysis::ActionReport& action : node.actions)
    {
        jsonNode["actions"].push_back({
            {"kind", ActionCode(action)},
            {"amountTo", ToChipUnits(action.amountTo)},
            {"chipsCommitted", ToChipUnits(action.chipsCommitted)},
            {"isAllIn", action.isAllIn},
            {"nextNodeId", action.nextNodeId.Value()},
        });
    }
    return jsonNode;
}
} // namespace

ServiceRequest ParseServiceRequest(const std::string& jsonLine)
{
    ServiceRequest request;
    try
    {
        const Json json = Json::parse(jsonLine);
        if (!json.at("requestId").is_number_unsigned())
            throw std::invalid_argument("requestId must be a non-negative integer");
        request.requestId = json.at("requestId").get<std::uint64_t>();
        const auto command = json.at("command").get<std::string>();
        request.equity = command == "query_equity";
        if (command != "query_node" && !request.equity)
            throw std::invalid_argument("Unknown solver command");

        const Json& nodeId = json.at("nodeId");
        if (!nodeId.is_number_unsigned() || nodeId.get<std::uint64_t>() > std::numeric_limits<std::int32_t>::max())
            throw std::invalid_argument("nodeId must be a non-negative 32-bit integer");
        request.nodeId = game::NodeId(nodeId.get<std::int32_t>());
    }
    catch (const std::exception& error)
    {
        request.validationError = error.what();
    }
    return request;
}

std::string ServiceMessageToJson(const ServiceMessage& message)
{
    Json json;
    switch (message.kind)
    {
    case ServiceMessageKind::BuildingTree:
        json = {{"event", "building_tree"}, {"totalIterations", message.totalIterations}};
        break;
    case ServiceMessageKind::Solving:
        json = {
            {"event", "solving"},
            {"completedIterations", message.completedIterations},
            {"totalIterations", message.totalIterations},
        };
        break;
    case ServiceMessageKind::Ready:
        json = {
            {"event", "ready"},
            {"iterations", message.completedIterations},
            {"nodeCount", message.nodeCount},
            {"rootNodeId", message.rootNodeId.Value()},
        };
        break;
    case ServiceMessageKind::Failed:
        json = {{"event", "failed"}, {"message", message.text}};
        break;
    case ServiceMessageKind::QuerySucceeded:
        json = {{"requestId", message.requestId}, {"ok", true}};
        if (message.equity)
            json["equity"] = BuildEquityJson(*message.equity);
        else
            json["node"] = BuildNodeJson(*message.node);
        break;
    case ServiceMessageKind::QueryFailed:
        json = {{"requestId", message.requestId}, {"ok", false}, {"error", message.text}};
        break;
    }
    return json.dump();
}
} // namespace solver::service
