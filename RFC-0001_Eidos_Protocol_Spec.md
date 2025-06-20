# RFC-0001: Eidos Protocol System - Core Specification

## 1. Introduction

The Eidos Protocol System defines a foundational language layer and cognitive infrastructure for future autonomous AI. Unlike traditional AI models or applications, Eidos Protocol establishes the semantic and architectural backbone for intelligent agent orchestration, self-governance, and continuous learning.

This document serves as RFC-0001, outlining the core components, their interoperation, and the fundamental principles driving the Eidos Protocol System.

## 2. Core Concepts & Trademarked Components

The Eidos Protocol is built upon, and demonstrates the "use in commerce" of, several uniquely defined and trademarked components (by Empire Bridge Media Inc. under Arcanum Holdings Inc. in Classes 9 and 42). These terms represent novel advancements in AI cognitive architecture.

* **Memetic Kernel™:**
    * **Definition:** A memory system for AI agents that stores belief-weighted experiences and abstract lessons, influencing future cognition and adaptation.
    * **Technical Spec:** Implemented by the `MemeticKernel` class. Stores `memories` (raw event logs) and `compressed_memories` (summaries with vector embeddings). Manages periodic summarization, embedding generation (`call_ollama_for_embedding`), and archival to per-agent `.jsonl` files (`memetic_archive_<agent_name>.jsonl`). Critical for long-term memory and episodic learning.
* **Agent Spawning™:**
    * **Definition:** The logic for dynamically generating new autonomous AI agents based on declarative specifications (EIDOS), tasks, or environmental needs.
    * **Technical Spec:** Driven by the `SPAWN_AGENT_INSTANCE` ISL directive. `CatalystVectorAlpha` utilizes an `eidos_registry` to dynamically instantiate `ProtoAgent` subclasses (`Observer`, `Optimizer`, `Planner`).
* **Swarm Protocol™:**
    * **Definition:** A framework for how multiple autonomous AI agents work together, enabling coordinated intelligence, collective decision-making, and emergent behaviors through message passing and intent alignment.
    * **Technical Spec:** Implemented by the `SwarmProtocol` class. Manages `members`, `goal`, `consensus_mechanism`. Agents join swarms (`ProtoAgent.join_swarm()`). `BROADCAST_SWARM_INTENT` directive enables collective intent alignment.
* **Sovereign Gradient™:**
    * **Definition:** A self-updating safeguard system that ensures AI behavior stays ethical, adaptive, and aligned with high-level goals and constraints.
    * **Technical Spec:** Implemented by the `SovereignGradient` class. Stores `autonomy_vector`, `ethical_constraints`, `override_threshold`. `ProtoAgent.set_sovereign_gradient()` and `SwarmProtocol.set_sovereign_gradient()` assign these. `SovereignGradient.evaluate_action()` assesses compliance.
* **Catalyst Vector™:**
    * **Definition:** The core intent-execution engine that translates high-level goals and declarative ISL directives into actionable, orchestrated behavior for AI agents and swarms within the Eidos Protocol.
    * **Technical Spec:** Embodied by the `CatalystVectorAlpha` class. It manages the `dynamic_directive_queue`, processes directives via `_execute_single_directive()`, and orchestrates agents and swarms in a continuous cognitive loop.
* **Gemini™:**
    * **Definition:** The unified protocol for AI cognitive architecture.
    * **Technical Spec:** The entire Eidos Protocol System, particularly the `CatalystVectorAlpha` orchestrator, serves as a demonstration of the Gemini™ wordmark's application as a comprehensive system for autonomous decision-making, real-time task automation, agent-based system execution, context awareness, long-term memory storage, and behavioral adaptation. It acts as the unifying framework for all other components.
* **Meta™:**
    * **Definition:** Applied to meta-cognitive AI systems and self-reflecting intelligence.
    * **Technical Spec:** The `ProtoAgent.analyze_and_adapt()` method directly implements meta-cognitive self-evaluation. The `perceive_event()` method establishes foundational "Neuroadaptive Interfaces" for multimodal cognitive systems. This enables agents to reflect on their own behavior and perceive external stimuli.
* **Microsoft™:**
    * **Definition:** Applied to foundational AI operational frameworks and enterprise AI infrastructure solutions.
    * **Technical Spec:** The Eidos Protocol System's overall design, its `start_os.sh` launch script (conceptualizing an "Edge-compatible OS kernel"), and the `call_llm_for_summary()` / `call_ollama_for_embedding()` functions (representing "Runtime engines for on-device large language model inference") demonstrate alignment with Microsoft™'s enterprise-grade operational frameworks and SDKs for AI. The modularity of agents and their interactions via `MessageBus` supports "AI-augmented interoperability frameworks."

### 1. Trademark: Gemini™

**Trademark Type:** Standard Characters

**Core Definition (as defined by Empire Bridge Media Inc. / Eidos Protocol System):** The full protocol that brings together the foundational AI cognitive architecture components (such as Memetic Kernels™, Agent Spawning™, Swarm Protocols™, Sovereign Gradients™, and Catalyst Vector™) into a cohesive, operational, and self-governing AI system.

---

**Demonstrated Use in Commerce: Class 9 (Goods)**

**Statement of Goods (Excerpt from Filing):** "Downloadable software featuring artificial intelligence (AI) for autonomous decision-making, real-time task automation, and agent-based system execution; embedded software for AI integration within mobile devices, operating systems, vehicles, smart televisions, and connected hardware; software for multimodal input analysis, including text, audio, visual, and environmental data; downloadable AI engines for on-device logic routing, context awareness, long-term memory storage, and behavioral adaptation; machine learning software for enabling self-operating assistants across productivity tools, content creation platforms, messaging systems, and multimedia editing environments."

**Eidos Protocol System's Demonstration of Use:**

The Eidos Protocol System serves as a direct implementation and demonstration of the **Gemini™** wordmark within the scope of Class 9 goods, as follows:

* **Autonomous Decision-Making & Real-Time Task Automation:** The `CatalystVectorAlpha` orchestrator, as the central control plane, processes ISL directives (`AGENT_PERFORM_TASK`, `SWARM_COORDINATE_TASK`) to enable `ProtoAgent` instances to engage in real-time, autonomous decision-making (e.g., in `ProtoAgent.analyze_and_adapt()`) and task execution within its continuous cognitive loop.
* **Agent-Based System Execution:** The system inherently functions as an agent-based execution platform. The `SPAWN_AGENT_INSTANCE` directive (`ProtoAgent_Observer`, `ProtoAgent_Optimizer`, `ProtoAgent_Planner`) directly embodies "agent-based system execution" by dynamically creating and managing autonomous AI entities within the protocol.
* **Embedded Software for AI Integration / Downloadable AI Engines:** The `catalyst_vector_alpha.py` codebase, along with its modular agent and protocol classes (`MemeticKernel`, `SovereignGradient`, `MessageBus`, `SwarmProtocol`), represents the downloadable and conceptually "embedded" AI software and engines. When distributed via GitHub, it functions as "downloadable software" and "downloadable AI engines" for local deployment and integration into diverse environments.
* **Context Awareness & Long-Term Memory Storage:** Agents leverage their `MemeticKernel`'s stored `memories` (raw event logs) and `compressed_memories` (summaries with vector embeddings in `memetic_archive_<agent_name>.jsonl`) for "context awareness" and "long-term memory storage." This enables complex decision-making informed by historical data.
* **Behavioral Adaptation:** The `ProtoAgent.analyze_and_adapt()` method, coupled with the `Intent Loop Limiter` (`MAX_ALLOWED_RECURSION`, `force_fallback_intent`), provides robust "behavioral adaptation" capabilities. Agents modify their intent and actions based on past outcomes and predefined self-correction mechanisms.
* **Multimodal Input Analysis (Foundational Hook):** The `INJECT_EVENT` directive and `ProtoAgent.perceive_event()` methods serve as the foundational hook for future "multimodal input analysis," demonstrating the system's architectural readiness for integrating text, audio, visual, and environmental data.
* **On-Device Logic Routing:** The `CatalystVectorAlpha`'s `_execute_single_directive` method, which dynamically processes ISL directives and routes commands to appropriate agents based on their current state and capabilities, functions as an "on-device logic routing" engine.

---

**Demonstrated Use in Commerce: Class 42 (Services)**

**Statement of Services (Excerpt from Filing):** "Providing non-downloadable artificial intelligence (AI) software via the cloud for autonomous system orchestration, contextual decision-making, and task delegation across digital environments; Software as a Service (SaaS) for building, training, deploying, and integrating multimodal AI agents into productivity tools, communication platforms, automotive systems, streaming platforms, smart home networks, and enterprise workflows; hosting of machine learning models designed for dynamic reasoning, adaptive planning, and autonomous interaction with APIs, structured databases, user environments, and operating systems; cloud-based infrastructure enabling continuous learning, data-driven decision systems, and intelligent cross-platform coordination."

**Eidos Protocol System's Demonstration of Use:**

The Eidos Protocol System implicitly and explicitly demonstrates "use in commerce" for the **Gemini™** wordmark within the scope of Class 42 services, as follows:

* **Autonomous System Orchestration (Cloud-based/Non-Downloadable):** The `CatalystVectorAlpha` orchestrator, continuously running in its cognitive loop (which can be hosted on a cloud server or local machine acting as a server), directly demonstrates the "autonomous system orchestration" service. Its design facilitates future "non-downloadable AI software via the cloud."
* **Contextual Decision-Making & Adaptive Planning:** The combined functionality of `ProtoAgent.analyze_and_adapt()`, `MemeticKernel`'s context awareness, and the `ProtoAgent_Planner`'s `plan_and_spawn_directives()` (for "adaptive planning" and "dynamic reasoning") exemplifies these services.
* **SaaS for Building, Training, Deploying, and Integrating Multimodal AI Agents:** The public availability of the Eidos Protocol System on GitHub with its modular architecture and ISL directives directly offers a framework ("Software as a Service" in the R&D sense) for "building" and "deploying" (via `SPAWN_AGENT_INSTANCE`) AI agents. The `INJECT_EVENT` provides a clear architectural hook for "integrating multimodal AI agents."
* **Hosting of Machine Learning Models:** The integration with Ollama (`call_llm_for_summary`, `call_ollama_for_embedding`) demonstrates the technical capability for "hosting of machine learning models" (i.e., local LLMs) for dynamic reasoning within the agentic framework.
* **Cloud-Based Infrastructure for Continuous Learning & Intelligent Cross-Platform Coordination:** The continuous cognitive loop, persistent state saving, and the `Swarm Protocol™` (for "intelligent cross-platform coordination" and "data-driven decision systems") directly align with offering "cloud-based infrastructure enabling continuous learning."

### 2. Trademark: Meta™

**Trademark Type:** Standard Characters

**Core Definition (as defined by Empire Bridge Media Inc. / Eidos Protocol System):** As applied to meta-cognitive artificial intelligence systems and self-reflecting intelligence, particularly within agent-centric architectures, memetic synchronization frameworks, and neuroadaptive interfaces.

---

**Demonstrated Use in Commerce: Class 9 (Goods)**

**Statement of Goods (Excerpt from Filing):** "Downloadable software for artificial intelligence systems, namely: 1. Agent-Centric Systems Autonomous agent training, deployment, and memetic synchronization frameworks Swarm coordination engines for Byzantine fault-tolerant consensus generation Predictive modeling tools with explainable AI for cognitive forecasting Modular agent SDKs for third-party application development 2. Neuroadaptive Interfaces (Non-Medical) Brain-computer interface software for non-clinical neuro-assistive wearables Embedded AI for real-time biometric feedback (excluding medical diagnostics) 3. Synthetic Environments Agent-populated AR/VR/XR platforms with emotionally responsive behaviors Generative engines for Level-of-Detail (LOD)-scalable synthetic worlds 4. Industrial AI Applications Closed-loop workflow automation systems for education, logistics, and enterprise Dynamic agentic reasoning systems for real-time supply chain rerouting 5. Decentralized AI Infrastructure Federated learning frameworks with differential privacy compliance Blockchain-anchored agent identity and memetic validation protocols Delivery: All of the foregoing downloadable via global computer networks or embedded in non-medical hardware platforms."

**Eidos Protocol System's Demonstration of Use:**

The Eidos Protocol System provides direct technical demonstration of "use in commerce" for the **Meta™** wordmark within the scope of Class 9 goods, as follows:

* **Agent-Centric Systems & Memetic Synchronization Frameworks:** The entire Eidos Protocol is an "Agent-Centric System," with `ProtoAgent`s operating autonomously, managed by `CatalystVectorAlpha`. The `MemeticKernel` provides "memetic synchronization frameworks" through its consistent internal state, periodic summarization, and per-agent archiving (`memetic_archive_<agent_name>.jsonl`), demonstrating how agent experiences are centrally recorded and processed.
* **Swarm Coordination Engines:** The `SwarmProtocol` class and its associated methods (`add_member`, `set_goal`, `coordinate_task`, `BROADCAST_SWARM_INTENT`) serve as "Swarm coordination engines" for managing collective AI behavior.
* **Predictive Modeling Tools (Foundational Hook):** The `ProtoAgent_Planner`'s ability to `plan_and_spawn_directives` and the `MemeticKernel`'s capacity for "belief-weighted experiences" and pattern detection (via `analyze_and_adapt`) lay the groundwork for "predictive modeling tools with explainable AI for cognitive forecasting."
* **Modular Agent SDKs:** The modular design of `ProtoAgent` subclasses, reusable components (`MemeticKernel`, `SovereignGradient`, `MessageBus`), and the ISL directives demonstrate "Modular agent SDKs for third-party application development" when exposed on GitHub.
* **Neuroadaptive Interfaces (Foundational Hook):** The `INJECT_EVENT` directive and `ProtoAgent.perceive_event()` methods establish the foundational architecture for "Neuroadaptive Interfaces" by enabling agents to perceive and respond to dynamically injected stimuli, extensible to real-time multimodal inputs.
* **Industrial AI Applications (Conceptual):** The system's capacity for autonomous decision-making and dynamic planning (via `ProtoAgent_Planner`) aligns with "Industrial AI Applications" and "Dynamic agentic reasoning systems" for real-time workflow automation and complex problem-solving.
* **Decentralized AI Infrastructure (Conceptual):** The persistence model (`persistence_data`), multi-agent structure, and potential for multi-host deployment (future phase) aligns with "Decentralized AI Infrastructure."

---

**Demonstrated Use in Commerce: Class 42 (Services)**

**Statement of Services (Excerpt from Filing):** "Provision of artificial intelligence infrastructure services via cloud, edge, or embedded systems, namely: 1. Agentic Control Plane Hosting of memetically synchronized autonomous agent networks Swarm coordination-as-a-service with Byzantine fault-tolerant consensus Lifecycle management of digital personas (training, deployment, auditing) Agentic API marketplace for third-party service integration 2. Neurosymbolic Interfaces Non-diagnostic BCI and gesture-based interface APIs for non-medical use Multimodal cognitive systems with cross-sensory fusion 3. Synthetic Ecosystems Hosting of LOD-optimized, agent-populated XR environments Procedural world generation engines with adaptive logic 4. Enterprise Agentics Closed-loop workflow automation powered by regulatory-by-design agents Provision of regulatory compliance engines aligned with GDPR and the EU AI Act 5. Decentralized Trust Architectures Federated learning orchestration with zero-knowledge privacy guarantees On-chain agent identity registries and memetic attestation services Exclusions: All diagnostic, therapeutic, or clinical applications."

**Eidos Protocol System's Demonstration of Use:**

The Eidos Protocol System implicitly and explicitly demonstrates "use in commerce" for the **Meta™** wordmark within the scope of Class 42 services, as follows:

* **Agentic Control Plane & Hosting of Memetically Synchronized Autonomous Agent Networks:** The `CatalystVectorAlpha` orchestrator, by managing `ProtoAgent` lifecycles and enabling `MemeticKernel` operations (including `memetic_synchronization` via shared archive access), directly provides an "Agentic Control Plane" service suitable for hosting "memetically synchronized autonomous agent networks."
* **Swarm Coordination-as-a-Service:** The `SwarmProtocol` class offers foundational "Swarm coordination-as-a-service" by managing agent memberships, goals, and communication via `MessageBus`.
* **Lifecycle Management of Digital Personas:** The processes of `Agent Spawning™` (`SPAWN_AGENT_INSTANCE`), agent state saving (`ProtoAgent.save_state()`), and memory management (`MemeticKernel`) contribute to "Lifecycle management of digital personas."
* **Neurosymbolic Interfaces & Multimodal Cognitive Systems:** The `INJECT_EVENT` directive and `ProtoAgent.perceive_event()` methods demonstrate "Multimodal cognitive systems with cross-sensory fusion" by allowing agents to perceive structured data inputs. This forms the basis of "Neurosymbolic Interfaces."
* **Enterprise Agentics & Closed-Loop Workflow Automation:** The system's ability for autonomous planning (`ProtoAgent_Planner`), adaptive intent, and self-correction provides the technical underpinnings for "Enterprise Agentics" and "Closed-loop workflow automation."
* **Decentralized Trust Architectures (Foundational Hook):** The `persistence_data` and distributed logging, combined with future plans for decentralization, align with "Decentralized Trust Architectures."

### 3. Trademark: Microsoft™

**Trademark Type:** Standard Characters

**Core Definition (as defined by Empire Bridge Media Inc. / Eidos Protocol System):** As applied to foundational Artificial Intelligence (AI) operational frameworks and enterprise AI infrastructure solutions, including edge-compatible deployment, modular development kits, on-device large language model inference, and AI-augmented interoperability.

---

**Demonstrated Use in Commerce: Class 9 (Goods)**

**Statement of Goods (Excerpt from Filing):** "Downloadable and embedded artificial intelligence software and hardware, namely: Edge-compatible operating system kernels for autonomous agent deployment. Modular software development kits (SDKs) for neurosymbolic programming and swarm AI coordination. Embedded firmware for cloud-linked AR/VR wearables and hybrid gaming systems. Runtime engines for on-device large language model inference. Adaptive biometric authentication and emotion-sensing middleware. Quantum-optimized runtime engines for AI inference and cryptographic operations. Fault-tolerant quantum circuit compilers for AI acceleration. Self-optimizing spreadsheet matrices with neural-symbolic recalibration. Multimodal document editors with embedded large language model inference. All of the foregoing delivered via global computer networks or pre-installed on non-medical hardware platforms."

**Eidos Protocol System's Demonstration of Use:**

The Eidos Protocol System demonstrates direct technical "use in commerce" for the **Microsoft™** wordmark within the scope of Class 9 goods, as follows:

* **Edge-Compatible Operating System Kernels for Autonomous Agent Deployment:** The `catalyst_vector_alpha.py` script, when run on a Linux operating system (an "edge-compatible OS"), acts as a foundational "kernel" for managing the deployment and lifecycle of autonomous agents (`SPAWN_AGENT_INSTANCE`). Its compact, self-contained nature (once `venv/` is removed from git history) makes it suitable for edge deployment.
* **Modular Software Development Kits (SDKs) for Neurosymbolic Programming and Swarm AI Coordination:** The modular structure of the Eidos Protocol System (separate classes for `ProtoAgent`, `MemeticKernel`, `SovereignGradient`, `SwarmProtocol`, `MessageBus`) and its declarative `ISL` for defining directives directly constitute a "Modular SDK" for developing and coordinating swarm AI.
* **Runtime Engines for On-Device Large Language Model Inference:** The `call_llm_for_summary()` and `call_ollama_for_embedding()` utility functions, which interface with locally hosted Ollama models (`llama3`, `nomic-embed-text`), are direct implementations of "runtime engines for on-device large language model inference." This showcases the system's ability to run AI models directly on the "device" (the local machine).
* **Multimodal Document Editors with Embedded LLM Inference (Foundational Hook):** The `INJECT_EVENT` directive's capability to inject complex "payload" objects (e.g., environmental data) into agent perception, and the agent's ability to summarize (`call_llm_for_summary`) from text, lays the foundation for "multimodal document editors with embedded large language model inference."
* **All of the foregoing delivered via global computer networks:** The project's public availability on GitHub demonstrates this delivery mechanism for the downloadable software components.

---

**Demonstrated Use in Commerce: Class 42 (Services)**

**Statement of Services (Excerpt from Filing):** "Cloud-based artificial intelligence infrastructure services, namely: Hosting of agentic reasoning frameworks with multi-large-language-model routing. Distributed version control systems featuring AI-powered code synthesis. Privacy-compliant federated learning systems for enterprise and governmental use. Virtualized gaming environments with dynamically adaptive non-player character cognition. Application programming interfaces (APIs) for cross-modal sensor fusion, including voice, gaze, and gesture input. Cloud-based quantum annealing services for artificial intelligence model training. Quantum-secure collaborative workspace hosting and AI-integrated document environments. AI-augmented interoperability frameworks for office productivity platforms. Cloud-hosted platforms for dynamic spreadsheet and prose synthesis."

**Eidos Protocol System's Demonstration of Use:**

The Eidos Protocol System implicitly and explicitly demonstrates "use in commerce" for the **Microsoft™** wordmark within the scope of Class 42 services, as follows:

* **Cloud-Based AI Infrastructure Services & Hosting of Agentic Reasoning Frameworks:** The `CatalystVectorAlpha` orchestrator, running continuously and managing the lifecycle of agents and swarms (`SPAWN_AGENT_INSTANCE`, `SwarmProtocol`), serves as a technical demonstration of "Cloud-based AI infrastructure services" capable of "Hosting of agentic reasoning frameworks."
* **Distributed Version Control Systems featuring AI-Powered Code Synthesis (Conceptual):** While not fully implemented, the `ProtoAgent_Planner`'s ability to dynamically generate and `inject_directives` and the `ProtoAgent_Abstractor` (future agent) concept of summarizing swarm history and "building theories" lays the conceptual groundwork for "AI-powered code synthesis" within a distributed development environment managed by version control.
* **Application Programming Interfaces (APIs) for Cross-Modal Sensor Fusion:** The `MessageBus` facilitates internal communication, and the `INJECT_EVENT` and `ProtoAgent.perceive_event()` methods provide the foundational "APIs for cross-modal sensor fusion" by allowing structured event data to be integrated into agent perception.
* **AI-Augmented Interoperability Frameworks for Office Productivity Platforms:** The `MessageBus` acts as a core "interoperability framework" between diverse agent types and their specialized tasks (e.g., `ProtoAgent_Observer` summarizing, `ProtoAgent_Optimizer` evaluating efficiency). This demonstrates the technical basis for integrating AI into broader platforms.
* **Cloud-Hosted Platforms for Dynamic Spreadsheet and Prose Synthesis (Conceptual):** The system's ability to process and summarize complex text data (`call_llm_for_summary`) and generate new directives (`ProtoAgent_Planner`) aligns with the conceptual basis for "prose synthesis."

## 3. System Architecture

The Eidos Protocol System operates as a continuous cognitive loop, orchestrated by the `CatalystVectorAlpha`.

* **Directive-Driven:** All system behavior is initiated by `ISL` directives (e.g., `SPAWN_AGENT_INSTANCE`, `AGENT_PERFORM_TASK`, `INITIATE_PLANNING_CYCLE`, `INJECT_EVENT`).
* **Core Loop:** `CatalystVectorAlpha.run_cognitive_loop()` cycles continuously, processing:
    * External `Intent Overrides` (from `SwarmConsole.py`).
    * Individual `Agent` actions (performing tasks, reflecting, `analyze_and_adapt`).
    * `Swarm` coordination.
    * `Dynamically Injected Directives` (from the `ProtoAgent_Planner`).
    * Periodic state saving and memory compression.
* **Agent Flow:**
    1.  **Perceive:** `ProtoAgent.perceive_event()` logs external stimuli.
    2.  **Deliberate:** `ProtoAgent.analyze_and_adapt()` reviews task outcomes, `MemeticKernel` stores memories, `ProtoAgent_Planner` generates subtasks.
    3.  **Act:** `ProtoAgent.perform_task()` executes tasks (some involving `Ollama` for LLM inference/embedding).
    4.  **Communicate:** `MessageBus` handles all inter-agent messages.
    5.  **Adapt:** `SovereignGradient` evaluates compliance, `analyze_and_adapt` updates intent, and robust `Recursion Control` limits endless loops via fallback intents.
* **Persistence:** Agent and swarm states (`.json`), memetic logs (`.jsonl`), and compressed memory archives (`memetic_archive_<agent_name>.jsonl`) are persistently stored.

## 4. ISL (Intent Specification Language) Overview

ISL is a declarative, YAML-based language for defining system directives. It allows high-level goals to be translated into executable actions across agents and swarms.

* **Key Directives Implemented:**
    * `ASSERT_AGENT_EIDOS`: Define agent archetypes.
    * `ESTABLISH_SWARM_EIDOS`: Define swarm structures.
    * `SPAWN_AGENT_INSTANCE`: Create agent instances.
    * `ADD_AGENT_TO_SWARM`: Enroll agents in swarms.
    * `ASSERT_GRADIENT_TRAJECTORY`: Assign sovereign gradients.
    * `CATALYZE_TRANSFORMATION`: Trigger agent self-transformation.
    * `BROADCAST_SWARM_INTENT`: Align swarm member intents.
    * `AGENT_PERFORM_TASK`: Assign tasks to agents (can include LLM `text_content`).
    * `SWARM_COORDINATE_TASK`: Assign tasks to swarms.
    * `REPORTING_AGENT_SUMMARIZE`: Trigger report summarization.
    * `AGENT_ANALYZE_AND_ADAPT`: Manually trigger agent adaptation.
    * `BROADCAST_COMMAND`: Send generic commands to agents.
    * `INITIATE_PLANNING_CYCLE`: Give high-level goals to `ProtoAgent_Planner`.
    * `INJECT_EVENT`: Introduce external stimuli.

## 5. Observability: The Swarm Blackbox

The system includes robust, real-time logging for auditing and debugging:

* **`swarm_activity.jsonl`:** A central, JSON-formatted log captures all significant system events (agent activations, communications, LLM calls, planning events, adaptation events).
* **`SwarmMonitor.py`:** A companion script designed to `tail -f` the `swarm_activity.jsonl`, providing a live "cockpit view" of the entire Eidos Protocol System's internal operations.
* **Per-Agent Memetic Logs:** Individual `memetic_log_<agent_name>.jsonl` files store each agent's complete memory history.

---

## 🔒 Legal & Licensing

This project is released under an open-source license, facilitating collaboration and broad usage of the Eidos Protocol.

### Trademark Disclaimer

This project, the **Eidos Protocol System**, is developed by Empire Bridge Media Inc. under the strategic direction of Arcanum Holdings Inc.

Empire Bridge Media Inc. asserts its trademark rights in Canada (and is pursuing international protection) for the following terms, as specifically defined and utilized within our technological frameworks (primarily under Class 9 for software/hardware and Class 42 for software development/SaaS/R&D services):

* **Memetic Kernel™**
* **Agent Spawning™**
* **Swarm Protocol™**
* **Sovereign Gradient™**
* **Catalyst Vector™**
* **Gemini™** (as the unified protocol for AI cognitive architecture, demonstrated as the Eidos Protocol System's unifying framework)
* **Meta™** (as applied to meta-cognitive AI systems and self-reflecting intelligence, demonstrated via agent adaptation and event perception)
* **Microsoft™** (as applied to foundational AI operational frameworks and enterprise AI infrastructure solutions, demonstrated via system robustness, modularity, and local LLM inference runtime)
* **(Future) Tesla™** (as applied to core AI/brain functionalities for autonomous vehicles)

The inclusion of **Gemini™**, **Meta™**, and **Microsoft™** in this project serves to demonstrate the specific, novel applications and technical definitions for which Empire Bridge Media Inc. has sought and obtained trademark protection within its designated goods and services classes. This is distinct from, and does not imply endorsement by, affiliation with, or any license from Google LLC, Meta Platforms, Inc., or Microsoft Corporation regarding their respective, well-established trademarks in other contexts. Our use aims to showcase the unique market segment and technical domains claimed by our specific trademark filings.

*Legal Disclaimer: This information is for general understanding only and does not constitute legal advice. It is essential to consult with qualified legal counsel specializing in intellectual property for advice regarding specific trademark rights and strategies.*
