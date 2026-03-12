"""
UN SDG-3 AI Platform — Unified Drug Discovery & Healthcare Supply Chain System.
Orchestrates two interconnected pipelines:
  1. Drug discovery lead optimization (Llama-3 agent + KG + RDKit)
  2. Healthcare supply chain decision support (forecasting + OR-Tools VRP)
"""
from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agents.drug_discovery_agent import DrugDiscoveryAgent
from knowledge_graph.chem_protein_kg import ChemProteinKG
from knowledge_graph.supply_chain_kg import SupplyChainKG
from forecasting.stockout_forecaster import StockoutForecaster
from optimization.vrp_optimizer import VRPOptimizer
from optimization.lead_selector import LeadSelector


class PipelineMode(str, Enum):
    DRUG_DISCOVERY = "drug_discovery"
    SUPPLY_CHAIN = "supply_chain"
    UNIFIED = "unified"            # Both pipelines share KG infrastructure


class PlatformRequest(BaseModel):
    request_id: str
    mode: PipelineMode = PipelineMode.UNIFIED

    # Drug discovery fields
    target_protein: str = ""
    seed_smiles: list[str] = Field(default_factory=list)
    optimization_objectives: list[str] = Field(default_factory=list)

    # Supply chain fields
    facility_ids: list[str] = Field(default_factory=list)
    forecast_horizon_days: int = 30
    demand_spike_factor: float = 1.0


class PlatformResponse(BaseModel):
    request_id: str
    mode: PipelineMode

    # Drug discovery outputs
    optimized_leads: list[dict] = Field(default_factory=list)
    kg_reasoning_trace: list[str] = Field(default_factory=list)
    task_success_rate: float = 0.0

    # Supply chain outputs
    stockout_risk_zones: list[dict] = Field(default_factory=list)
    rerouting_plans: list[dict] = Field(default_factory=list)
    flagged_anomalies: list[str] = Field(default_factory=list)

    # Shared audit log
    audit_log: list[dict] = Field(default_factory=list)


class UNSDG3Platform:
    """
    Unified AI platform for UN SDG-3 (Good Health and Well-Being).
    Shares Neo4j KG infrastructure and AWS SageMaker deployment
    across drug discovery and supply chain pipelines.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        # Shared KG infrastructure (chemical-protein + supply chain nodes)
        self.chem_kg = ChemProteinKG(cfg["chem_kg"])
        self.supply_kg = SupplyChainKG(cfg["supply_kg"])

        # Drug discovery pipeline
        self.drug_agent = DrugDiscoveryAgent(cfg["drug_discovery"], self.chem_kg)

        # Supply chain pipeline
        self.forecaster = StockoutForecaster(cfg["forecasting"])
        self.vrp_optimizer = VRPOptimizer(cfg["optimization"])
        self.lead_selector = LeadSelector(cfg["lead_selection"])

    async def run(self, request: PlatformRequest) -> PlatformResponse:
        response = PlatformResponse(request_id=request.request_id, mode=request.mode)

        if request.mode in (PipelineMode.DRUG_DISCOVERY, PipelineMode.UNIFIED):
            dd_result = await self._run_drug_discovery(request)
            response.optimized_leads = dd_result["leads"]
            response.kg_reasoning_trace = dd_result["reasoning"]
            response.task_success_rate = dd_result["task_success_rate"]
            response.audit_log.extend(dd_result["audit"])

        if request.mode in (PipelineMode.SUPPLY_CHAIN, PipelineMode.UNIFIED):
            sc_result = await self._run_supply_chain(request)
            response.stockout_risk_zones = sc_result["risk_zones"]
            response.rerouting_plans = sc_result["rerouting_plans"]
            response.flagged_anomalies = sc_result["anomalies"]
            response.audit_log.extend(sc_result["audit"])

        return response

    async def _run_drug_discovery(self, request: PlatformRequest) -> dict:
        # Step 1: KG multi-hop reasoning for target/off-target analysis
        kg_context = await self.chem_kg.multi_hop_query(
            target=request.target_protein,
            seed_smiles=request.seed_smiles,
        )
        # Step 2: Llama-3 agent optimizes leads with plan-and-verify
        agent_result = await self.drug_agent.optimize_leads(
            seed_smiles=request.seed_smiles,
            target=request.target_protein,
            kg_context=kg_context,
            objectives=request.optimization_objectives,
        )
        # Step 3: Multi-objective lead selection with SA score thresholds
        selected_leads = self.lead_selector.select(
            candidates=agent_result["candidates"],
            objectives=request.optimization_objectives,
        )
        return {
            "leads": selected_leads,
            "reasoning": agent_result["reasoning_trace"],
            "task_success_rate": agent_result["task_success_rate"],
            "audit": agent_result["audit_log"],
        }

    async def _run_supply_chain(self, request: PlatformRequest) -> dict:
        # Step 1: Forecast stockout risk per H3 zone
        risk_zones = await self.forecaster.forecast(
            facility_ids=request.facility_ids,
            horizon_days=request.forecast_horizon_days,
            demand_spike_factor=request.demand_spike_factor,
        )
        # Step 2: Query KG for anomalies and bottlenecks
        anomalies = await self.supply_kg.detect_anomalies(
            facility_ids=request.facility_ids,
        )
        # Step 3: Generate rerouting plans with OR-Tools VRP
        rerouting = await self.vrp_optimizer.solve(
            risk_zones=risk_zones,
            anomalies=anomalies,
            demand_spike_factor=request.demand_spike_factor,
        )
        return {
            "risk_zones": risk_zones,
            "rerouting_plans": rerouting,
            "anomalies": anomalies,
            "audit": [{"step": "supply_chain_complete", "zones": len(risk_zones)}],
        }
