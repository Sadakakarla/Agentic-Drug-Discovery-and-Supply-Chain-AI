# UN SDG-3 — Drug Discovery & Healthcare Supply Chain

> An end-to-end AI platform for UN Sustainable Development Goal 3 (Good Health and Well-Being), combining autonomous drug discovery lead optimization with intelligent healthcare supply chain decision support — built as a Millennium Fellow initiative under United Nations Academic Impact.

---

## Overview

Access to effective medicines and reliable healthcare supply chains are two of the most critical barriers to global health equity. This platform addresses both simultaneously through a unified AI system that shares knowledge graph infrastructure across two interconnected pipelines:

**Pipeline (i) — Drug Discovery Lead Optimization:** 

A tool-augmented Llama-3-8B agent that autonomously optimizes molecular lead compounds for disease targets, enforcing lab-feasibility through RDKit property checks and synthetic accessibility thresholds, with multi-hop reasoning over a 50K-node chemical-protein knowledge graph.

**Pipeline (ii) — Healthcare Supply Chain Decision Support:** 

A real-time decision system that forecasts zone-level stockout risk using GraphSAGE + XGBoost over facility-network signals, detects inventory anomalies via a Neo4j knowledge graph built from 4K+ requisition logs, and generates optimal rerouting plans using OR-Tools VRP under simulated demand spikes.

---

## What does this solve?

Traditional drug discovery pipelines rely on manual expert review of molecular candidates, leading to slow iteration cycles and high rates of infeasible compounds entering synthesis queues. Simultaneously, healthcare supply chains in resource-limited settings lack predictive tools to anticipate and respond to stockouts before they impact patient care.

This platform automates both workflows end-to-end: the drug discovery agent uses plan-and-verify reasoning to self-reject invalid candidates before they reach the lab, while the supply chain system provides proactive, geospatially-indexed risk scores and actionable rerouting plans before shortages occur — with full audit logs for compliance and human-in-the-loop review gates for critical decisions.

---

## Results

| Metric | Baseline | This System | Improvement |
|---|---|---|---|
| Drug Discovery Task Success Rate | 41% | **63%** | +22 pts |
| Invalid SMILES Generation Rate | ~35% | **~25%** | −28% |
| KG Query Failure Rate | ~40% | **~26%** | −35% |
| Retrieval Accuracy vs BioGPT | 68% | **78%** | +15% lift |
| Supply Chain Surge-Window RMSE | 24.5 (ARIMA) | **20.1** | −18% |
| Requisition Logs Structured | Manual | **4,000+** | Automated |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              UN SDG-3 AI Platform                    │
│                                                     │
│  ┌──────────────────┐   ┌───────────────────────┐  │
│  │  Drug Discovery  │   │   Supply Chain DSS    │  │
│  │                  │   │                       │  │
│  │  Llama-3-8B      │   │  DeBERTa-v3 NER       │  │
│  │  Plan+Verify     │   │  Requisition Parser   │  │
│  │  SFT+DPO+RS      │   │  GraphSAGE + XGBoost  │  │
│  │  RDKit Checks    │   │  H3 Zone Forecasting  │  │
│  │  SA Score Gate   │   │  OR-Tools VRP         │  │
│  └────────┬─────────┘   └──────────┬────────────┘  │
│           │                        │               │
│  ┌────────▼────────────────────────▼────────────┐  │
│  │     Shared Neo4j Knowledge Graph (50K nodes)  │  │
│  │   Chemical-Protein + Supply Chain Topology    │  │
│  │   DeBERTa-v3 Entity Linking                  │  │
│  │   Llama-2-7B Text-to-Cypher Interface        │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│         AWS SageMaker + Docker + vLLM               │
│         4-bit quantization + Audit Logs             │
└─────────────────────────────────────────────────────┘
```

---

## Key Technical Decisions

**plan-and-verify for drug discovery: Unconstrained LLM generation produces high rates of chemically invalid SMILES. The plan-and-verify loop — combined with rejection sampling across 3 rounds — filters invalid candidates before they consume synthesis resources, cutting invalid SMILES by 28%.

**GraphSAGE + XGBoost over ARIMA for stockout forecasting:** ARIMA models time series independently per facility and cannot capture network effects — a stockout at a distributor affects 12 downstream clinics simultaneously. GraphSAGE embeds each facility in the context of its supply network neighborhood before XGBoost classification, improving surge-window RMSE by 18%.

**OR-Tools for rerouting** OR-Tools CP-SAT/VRP provides provably optimal or near-optimal routing solutions with hard time-window constraints, unlike heuristic approaches that may violate critical delivery windows for high-priority facilities.

---

## Stack

**Agents:** Llama-3-8B (SFT + DPO + Rejection Sampling), LangChain, plan-and-verify reasoning

**Knowledge Graph:** Neo4j (50K nodes), DeBERTa-v3 entity linking, Llama-2-7B Text-to-Cypher, NetworkX

**Cheminformatics:** RDKit, SA Score, Lipinski RO5, QED

**Forecasting:** GraphSAGE (PyTorch Geometric), XGBoost, H3 geospatial indexing

**Optimization:** OR-Tools VRP with time windows, multi-objective lead selection

**Infrastructure:** AWS SageMaker, Docker, vLLM, 4-bit quantization, audit logging, human-in-the-loop review gates

---

## Quick Start

```bash
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run unified platform
python -c "
import asyncio, yaml
from pipeline.platform import UNSDG3Platform, PlatformRequest, PipelineMode
cfg = yaml.safe_load(open('configs/default.yaml'))
platform = UNSDG3Platform(cfg)
req = PlatformRequest(
    request_id='demo_001',
    mode=PipelineMode.UNIFIED,
    target_protein='EGFR',
    seed_smiles=['CC1=CC=CC=C1'],
    facility_ids=['facility_001', 'facility_002'],
    forecast_horizon_days=30,
)
result = asyncio.run(platform.run(req))
print(f'Leads: {len(result.optimized_leads)}, Risk Zones: {len(result.stockout_risk_zones)}')
"
```

---

## Project Structure

```
├── pipeline/               # Unified platform orchestrator
├── agents/                 # Llama-3 drug discovery agent (SFT+DPO+RS)
├── knowledge_graph/        # Chemical-protein KG + supply chain KG
├── forecasting/            # GraphSAGE + XGBoost stockout forecaster
├── optimization/           # OR-Tools VRP + multi-objective lead selector
├── tests/                  # Unit + regression gate tests
└── configs/                # Platform configuration
```

---

*Built as part of the Millennium Fellowship under United Nations Academic Impact, aligned with SDG 3: Good Health and Well-Being.*
