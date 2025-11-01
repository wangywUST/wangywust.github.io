<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph-Contrastive Reasoning for E-Commerce Security - Detailed</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.2);
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 5px;
            font-size: 22px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 10px;
            font-size: 13px;
        }
        .authors {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
            font-size: 12px;
        }
        svg {
            width: 100%;
            height: auto;
        }
        .section-header {
            background: linear-gradient(90deg, #667eea, #764ba2);
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .tech-detail {
            background: #f8f9fa;
            border-left: 3px solid #667eea;
            padding: 10px;
            margin: 10px 0;
            font-size: 11px;
            line-height: 1.4;
        }
        .formula {
            background: #fff;
            border: 1px solid #dee2e6;
            padding: 8px;
            margin: 5px 0;
            font-family: 'Courier New', monospace;
            font-size: 10px;
            text-align: center;
            border-radius: 4px;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(1.05); }
        }
        @keyframes flow {
            0% { stroke-dashoffset: 0; }
            100% { stroke-dashoffset: -20; }
        }
        .animated-edge {
            animation: flow 2s linear infinite;
        }
        .node-pulse {
            animation: pulse 3s ease-in-out infinite;
        }
        .highlight-box {
            fill: none;
            stroke: #ff5252;
            stroke-width: 2;
            stroke-dasharray: 5,5;
            animation: pulse 2s ease-in-out infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Graph-Contrastive Reasoning for Detecting Data Leakage and Security Anomalies</h1>
        <p class="subtitle">A Unified Framework for AI-Powered E-Commerce Security</p>
        <p class="authors">PI: Yiwei Wang, UC Merced | Amazon Research Award Proposal 2025</p>
        
        <svg viewBox="0 0 1350 900" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#f8f9fa;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#e9ecef;stop-opacity:1" />
                </linearGradient>
                
                <linearGradient id="alertGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#ff5252;stop-opacity:0.8" />
                    <stop offset="100%" style="stop-color:#ff1744;stop-opacity:0.8" />
                </linearGradient>
                
                <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur in="SourceAlpha" stdDeviation="2"/>
                    <feOffset dx="1" dy="1" result="offsetblur"/>
                    <feComponentTransfer>
                        <feFuncA type="linear" slope="0.3"/>
                    </feComponentTransfer>
                    <feMerge> 
                        <feMergeNode/>
                        <feMergeNode in="SourceGraphic"/> 
                    </feMerge>
                </filter>

                <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
                    <polygon points="0 0, 8 3, 0 6" fill="#666" />
                </marker>
            </defs>

            <!-- Main Title Section -->
            <rect x="10" y="10" width="1330" height="40" rx="8" fill="#667eea" opacity="0.1"/>
            <text x="675" y="35" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">
                Problem: Sophisticated attacks exploit multi-hop relationships in e-commerce graphs with billions of nodes
            </text>

            <!-- Section 1: Heterogeneous Graph Representation -->
            <g id="graph-section">
                <rect x="10" y="60" width="320" height="380" rx="8" fill="url(#bgGradient)" stroke="#dee2e6" stroke-width="1"/>
                <rect x="10" y="60" width="320" height="30" rx="8" fill="#667eea" opacity="0.8"/>
                <text x="170" y="80" text-anchor="middle" font-size="12" font-weight="bold" fill="white">1. Heterogeneous Graph G = (V, E, Tv, Te)</text>
                
                <!-- Graph visualization -->
                <g transform="translate(50, 110)">
                    <!-- Users -->
                    <circle cx="40" cy="40" r="18" fill="#FF6B6B" filter="url(#shadow)" class="node-pulse"/>
                    <text x="40" y="44" text-anchor="middle" font-size="9" fill="white">User</text>
                    
                    <!-- Products -->
                    <circle cx="120" cy="40" r="18" fill="#4ECDC4" filter="url(#shadow)"/>
                    <text x="120" y="44" text-anchor="middle" font-size="9" fill="white">Product</text>
                    
                    <!-- Transactions -->
                    <circle cx="200" cy="40" r="18" fill="#45B7D1" filter="url(#shadow)"/>
                    <text x="200" y="44" text-anchor="middle" font-size="9" fill="white">Trans</text>
                    
                    <!-- API Calls -->
                    <circle cx="40" cy="120" r="18" fill="#96CEB4" filter="url(#shadow)"/>
                    <text x="40" y="124" text-anchor="middle" font-size="9" fill="white">API</text>
                    
                    <!-- AI Agents -->
                    <circle cx="120" cy="120" r="18" fill="#DDA0DD" filter="url(#shadow)"/>
                    <text x="120" y="124" text-anchor="middle" font-size="9" fill="white">Agent</text>
                    
                    <!-- Data Resources -->
                    <circle cx="200" cy="120" r="18" fill="#FFB74D" filter="url(#shadow)"/>
                    <text x="200" y="124" text-anchor="middle" font-size="9" fill="white">Data</text>
                    
                    <!-- Connections with labels -->
                    <line x1="58" y1="40" x2="102" y2="40" stroke="#666" stroke-width="1.5" stroke-dasharray="4,4" class="animated-edge"/>
                    <text x="80" y="35" text-anchor="middle" font-size="7" fill="#666">purchase</text>
                    
                    <line x1="140" y1="40" x2="180" y2="40" stroke="#666" stroke-width="1.5" stroke-dasharray="4,4" class="animated-edge"/>
                    <text x="160" y="35" text-anchor="middle" font-size="7" fill="#666">in_cart</text>
                    
                    <line x1="40" y1="58" x2="40" y2="102" stroke="#666" stroke-width="1.5" stroke-dasharray="4,4" class="animated-edge"/>
                    <text x="25" y="80" text-anchor="middle" font-size="7" fill="#666">calls</text>
                    
                    <line x1="55" y1="55" x2="105" y2="105" stroke="#666" stroke-width="1.5" stroke-dasharray="4,4" class="animated-edge"/>
                    <text x="80" y="80" text-anchor="middle" font-size="7" fill="#666">delegates</text>
                    
                    <line x1="138" y1="120" x2="182" y2="120" stroke="#666" stroke-width="1.5" stroke-dasharray="4,4" class="animated-edge"/>
                    <text x="160" y="115" text-anchor="middle" font-size="7" fill="#666">access</text>
                    
                    <!-- Attack path highlight -->
                    <rect x="15" y="95" width="210" height="50" class="highlight-box"/>
                    <text x="120" y="165" text-anchor="middle" font-size="8" fill="#ff5252" font-weight="bold">Potential Attack Path</text>
                </g>
                
                <!-- Technical details -->
                <text x="20" y="290" font-size="10" font-weight="bold" fill="#333">Node Types (|V| ~ 10⁹):</text>
                <text x="20" y="305" font-size="9" fill="#666">• Users: Customer profiles, behavior patterns</text>
                <text x="20" y="318" font-size="9" fill="#666">• Products: Items, categories, inventory</text>
                <text x="20" y="331" font-size="9" fill="#666">• Transactions: Orders, payments, returns</text>
                <text x="20" y="344" font-size="9" fill="#666">• API Calls: Service requests, parameters</text>
                <text x="20" y="357" font-size="9" fill="#666">• Agents: AI assistants, automation bots</text>
                <text x="20" y="370" font-size="9" fill="#666">• Data Resources: DBs, files, credentials</text>
                
                <text x="20" y="390" font-size="10" font-weight="bold" fill="#333">Edge Types (|E| ~ 10¹⁰):</text>
                <text x="20" y="405" font-size="9" fill="#666">• Purchases, views, clicks, reviews</text>
                <text x="20" y="418" font-size="9" fill="#666">• API invocations, authentications</text>
                <text x="20" y="431" font-size="9" fill="#666">• Agent delegations, data accesses</text>
            </g>

            <!-- Section 2: Hierarchical GNN Architecture -->
            <g id="gnn-section">
                <rect x="340" y="60" width="320" height="380" rx="8" fill="url(#bgGradient)" stroke="#dee2e6" stroke-width="1"/>
                <rect x="340" y="60" width="320" height="30" rx="8" fill="#667eea" opacity="0.8"/>
                <text x="500" y="80" text-anchor="middle" font-size="12" font-weight="bold" fill="white">2. Hierarchical Graph Neural Network</text>
                
                <!-- Network architecture -->
                <g transform="translate(380, 110)">
                    <!-- Input layer -->
                    <text x="120" y="10" text-anchor="middle" font-size="9" fill="#666" font-weight="bold">Input Features X</text>
                    <circle cx="60" cy="30" r="10" fill="#FFD93D" filter="url(#shadow)"/>
                    <circle cx="90" cy="30" r="10" fill="#FFD93D" filter="url(#shadow)"/>
                    <circle cx="120" cy="30" r="10" fill="#FFD93D" filter="url(#shadow)"/>
                    <circle cx="150" cy="30" r="10" fill="#FFD93D" filter="url(#shadow)"/>
                    <circle cx="180" cy="30" r="10" fill="#FFD93D" filter="url(#shadow)"/>
                    
                    <!-- Intra-type message passing -->
                    <text x="120" y="70" text-anchor="middle" font-size="9" fill="#666" font-weight="bold">Intra-type MP</text>
                    <rect x="50" y="75" width="140" height="25" rx="3" fill="#6C5CE7" opacity="0.3"/>
                    <text x="120" y="92" text-anchor="middle" font-size="8" fill="#333">h_i^(l+1) = σ(W_intra Σ_j∈N_same h_j^(l))</text>
                    
                    <circle cx="75" cy="87" r="10" fill="#6C5CE7" filter="url(#shadow)"/>
                    <circle cx="120" cy="87" r="10" fill="#6C5CE7" filter="url(#shadow)"/>
                    <circle cx="165" cy="87" r="10" fill="#6C5CE7" filter="url(#shadow)"/>
                    
                    <!-- Inter-type message passing -->
                    <text x="120" y="130" text-anchor="middle" font-size="9" fill="#666" font-weight="bold">Inter-type MP</text>
                    <rect x="50" y="135" width="140" height="25" rx="3" fill="#00B894" opacity="0.3"/>
                    <text x="120" y="152" text-anchor="middle" font-size="8" fill="#333">h_i^(l+1) = σ(W_inter Σ_j∈N_diff h_j^(l))</text>
                    
                    <circle cx="75" cy="147" r="10" fill="#00B894" filter="url(#shadow)"/>
                    <circle cx="120" cy="147" r="10" fill="#00B894" filter="url(#shadow)"/>
                    <circle cx="165" cy="147" r="10" fill="#00B894" filter="url(#shadow)"/>
                    
                    <!-- Temporal aggregation -->
                    <text x="120" y="190" text-anchor="middle" font-size="9" fill="#666" font-weight="bold">Temporal Attention</text>
                    <rect x="50" y="195" width="140" height="25" rx="3" fill="#FF6B6B" opacity="0.3"/>
                    <text x="120" y="212" text-anchor="middle" font-size="8" fill="#333">α_t = softmax(q^T tanh(W_t h_t))</text>
                    
                    <circle cx="90" cy="207" r="10" fill="#FF6B6B" filter="url(#shadow)"/>
                    <circle cx="150" cy="207" r="10" fill="#FF6B6B" filter="url(#shadow)"/>
                    
                    <!-- Output embeddings -->
                    <text x="120" y="250" text-anchor="middle" font-size="9" fill="#666" font-weight="bold">Graph Embeddings H</text>
                    <circle cx="120" cy="270" r="12" fill="#FD79A8" filter="url(#shadow)"/>
                    
                    <!-- Connections -->
                    <line x1="120" y1="40" x2="120" y2="75" stroke="#999" stroke-width="1" opacity="0.5" marker-end="url(#arrowhead)"/>
                    <line x1="120" y1="100" x2="120" y2="135" stroke="#999" stroke-width="1" opacity="0.5" marker-end="url(#arrowhead)"/>
                    <line x1="120" y1="160" x2="120" y2="195" stroke="#999" stroke-width="1" opacity="0.5" marker-end="url(#arrowhead)"/>
                    <line x1="120" y1="220" x2="120" y2="258" stroke="#999" stroke-width="1" opacity="0.5" marker-end="url(#arrowhead)"/>
                </g>
                
                <!-- Architecture details -->
                <text x="350" y="390" font-size="10" font-weight="bold" fill="#333">Key Innovations:</text>
                <text x="350" y="405" font-size="9" fill="#666">• Hierarchical message passing captures</text>
                <text x="355" y="418" font-size="9" fill="#666">  entity-specific & cross-entity patterns</text>
                <text x="350" y="431" font-size="9" fill="#666">• Temporal attention prioritizes recent events</text>
            </g>

            <!-- Section 3: Contrastive Learning -->
            <g id="contrastive-section">
                <rect x="670" y="60" width="330" height="380" rx="8" fill="url(#bgGradient)" stroke="#dee2e6" stroke-width="1"/>
                <rect x="670" y="60" width="330" height="30" rx="8" fill="#667eea" opacity="0.8"/>
                <text x="835" y="80" text-anchor="middle" font-size="12" font-weight="bold" fill="white">3. Self-Supervised Contrastive Learning</text>
                
                <!-- Contrastive learning visualization -->
                <g transform="translate(680, 100)">
                    <!-- Positive pairs -->
                    <text x="70" y="20" text-anchor="middle" font-size="10" font-weight="bold" fill="#4CAF50">Positive Pairs</text>
                    <rect x="20" y="25" width="100" height="80" rx="5" fill="#4CAF50" opacity="0.2"/>
                    <text x="70" y="45" text-anchor="middle" font-size="8" fill="#333">Original Graph G_t</text>
                    <circle cx="50" cy="65" r="8" fill="#4CAF50" opacity="0.7"/>
                    <circle cx="70" cy="75" r="8" fill="#4CAF50" opacity="0.7"/>
                    <circle cx="90" cy="65" r="8" fill="#4CAF50" opacity="0.7"/>
                    <line x1="50" y1="65" x2="70" y2="75" stroke="#4CAF50" stroke-width="1"/>
                    <line x1="70" y1="75" x2="90" y2="65" stroke="#4CAF50" stroke-width="1"/>
                    <text x="70" y="95" text-anchor="middle" font-size="7" fill="#666">Benign augmentation</text>
                    
                    <!-- Negative samples -->
                    <text x="230" y="20" text-anchor="middle" font-size="10" font-weight="bold" fill="#F44336">Negative Samples</text>
                    <rect x="180" y="25" width="100" height="80" rx="5" fill="#F44336" opacity="0.2"/>
                    <text x="230" y="45" text-anchor="middle" font-size="8" fill="#333">Attack Simulation</text>
                    <circle cx="210" cy="65" r="8" fill="#F44336" opacity="0.7"/>
                    <circle cx="230" cy="75" r="8" fill="#F44336" opacity="0.7"/>
                    <circle cx="250" cy="65" r="8" fill="#F44336" opacity="0.7"/>
                    <line x1="210" y1="65" x2="250" y2="65" stroke="#F44336" stroke-width="2"/>
                    <line x1="210" y1="65" x2="230" y2="75" stroke="#F44336" stroke-width="2" stroke-dasharray="2,2"/>
                    <text x="230" y="95" text-anchor="middle" font-size="7" fill="#666">Injected anomalies</text>
                    
                    <!-- Embedding space -->
                    <text x="150" y="125" text-anchor="middle" font-size="10" font-weight="bold" fill="#333">Embedding Space</text>
                    <circle cx="90" cy="160" r="25" fill="#4CAF50" opacity="0.3"/>
                    <text x="90" y="165" text-anchor="middle" font-size="9" fill="#333">H⁺</text>
                    <circle cx="210" cy="160" r="25" fill="#F44336" opacity="0.3"/>
                    <text x="210" y="165" text-anchor="middle" font-size="9" fill="#333">H⁻</text>
                    
                    <line x1="115" y1="160" x2="185" y2="160" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
                    <text x="150" y="155" text-anchor="middle" font-size="8" fill="#666">maximize distance</text>
                    
                    <!-- Loss formula -->
                    <rect x="40" y="195" width="220" height="35" rx="3" fill="#fff" stroke="#dee2e6"/>
                    <text x="150" y="208" text-anchor="middle" font-size="9" fill="#333" font-weight="bold">Contrastive Loss:</text>
                    <text x="150" y="222" text-anchor="middle" font-size="8" fill="#666" font-family="monospace">
                        L = -log[exp(sim(H,H⁺)/τ) / Σexp(sim(H,H⁻)/τ)]
                    </text>
                </g>
                
                <!-- Attack simulation details -->
                <text x="680" y="350" font-size="10" font-weight="bold" fill="#333">Hard Negative Generation:</text>
                <text x="680" y="365" font-size="9" fill="#666">• Unauthorized access edge injection</text>
                <text x="680" y="378" font-size="9" fill="#666">• Temporal pattern anomalies (burst attacks)</text>
                <text x="680" y="391" font-size="9" fill="#666">• Confused deputy attacks simulation</text>
                <text x="680" y="404" font-size="9" fill="#666">• Multi-hop data exfiltration paths</text>
                <text x="680" y="417" font-size="9" fill="#666">• Privilege escalation chains</text>
                <text x="680" y="430" font-size="9" fill="#666">• API abuse patterns</text>
            </g>

            <!-- Section 4: LLM Integration -->
            <g id="llm-section">
                <rect x="1010" y="60" width="330" height="380" rx="8" fill="url(#bgGradient)" stroke="#dee2e6" stroke-width="1"/>
                <rect x="1010" y="60" width="330" height="30" rx="8" fill="#667eea" opacity="0.8"/>
                <text x="1175" y="80" text-anchor="middle" font-size="12" font-weight="bold" fill="white">4. LLM-Powered Explainability</text>
                
                <!-- LLM pipeline -->
                <g transform="translate(1020, 100)">
                    <!-- Step 1 -->
                    <rect x="10" y="10" width="140" height="40" rx="5" fill="#FF9800" opacity="0.8" filter="url(#shadow)"/>
                    <text x="80" y="25" text-anchor="middle" font-size="9" fill="white" font-weight="bold">1. Anomaly Detection</text>
                    <text x="80" y="40" text-anchor="middle" font-size="8" fill="white">Score: P(anomaly|G)</text>
                    
                    <!-- Step 2 -->
                    <rect x="170" y="10" width="140" height="40" rx="5" fill="#2196F3" opacity="0.8" filter="url(#shadow)"/>
                    <text x="240" y="25" text-anchor="middle" font-size="9" fill="white" font-weight="bold">2. Feature Extraction</text>
                    <text x="240" y="40" text-anchor="middle" font-size="8" fill="white">Subgraph → Text</text>
                    
                    <!-- Arrow -->
                    <line x1="150" y1="30" x2="170" y2="30" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
                    
                    <!-- Step 3 -->
                    <rect x="60" y="70" width="200" height="50" rx="5" fill="#9C27B0" opacity="0.8" filter="url(#shadow)"/>
                    <text x="160" y="88" text-anchor="middle" font-size="9" fill="white" font-weight="bold">3. LLM Analysis</text>
                    <text x="160" y="102" text-anchor="middle" font-size="8" fill="white">Context: {nodes, edges, patterns}</text>
                    <text x="160" y="114" text-anchor="middle" font-size="8" fill="white">→ Risk assessment & explanation</text>
                    
                    <!-- Arrows -->
                    <line x1="80" y1="50" x2="80" y2="70" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
                    <line x1="240" y1="50" x2="240" y2="70" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
                    
                    <!-- Step 4 -->
                    <rect x="60" y="140" width="200" height="60" rx="5" fill="#4CAF50" opacity="0.8" filter="url(#shadow)"/>
                    <text x="160" y="158" text-anchor="middle" font-size="9" fill="white" font-weight="bold">4. Human-Readable Report</text>
                    <text x="160" y="173" text-anchor="middle" font-size="7" fill="white">• Attack type classification</text>
                    <text x="160" y="185" text-anchor="middle" font-size="7" fill="white">• Impact severity (Critical/High/Med)</text>
                    <text x="160" y="197" text-anchor="middle" font-size="7" fill="white">• Mitigation recommendations</text>
                    
                    <line x1="160" y1="120" x2="160" y2="140" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
                    
                    <!-- Example prompt -->
                    <rect x="10" y="220" width="300" height="90" rx="3" fill="#f8f9fa" stroke="#dee2e6"/>
                    <text x="15" y="235" font-size="8" font-weight="bold" fill="#333">LLM Prompt Template:</text>
                    <text x="15" y="250" font-size="7" fill="#666" font-family="monospace">"Analyze security anomaly in e-commerce graph:</text>
                    <text x="15" y="262" font-size="7" fill="#666" font-family="monospace">Nodes: {user_123, agent_45, database_prod}</text>
                    <text x="15" y="274" font-size="7" fill="#666" font-family="monospace">Pattern: Unauthorized delegation chain</text>
                    <text x="15" y="286" font-size="7" fill="#666" font-family="monospace">Temporal: 50 requests in 2 seconds</text>
                    <text x="15" y="298" font-size="7" fill="#666" font-family="monospace">Assess risk and provide recommendations."</text>
                </g>
                
                <text x="1020" y="425" font-size="9" font-weight="bold" fill="#333">Natural Language Interface:</text>
                <text x="1020" y="438" font-size="8" fill="#666">Analysts can query: "Show all data access anomalies today"</text>
            </g>

            <!-- Bottom Section: Implementation Details -->
            <g id="implementation-section">
                <rect x="10" y="450" width="1330" height="440" rx="8" fill="url(#bgGradient)" stroke="#dee2e6" stroke-width="1"/>
                <rect x="10" y="450" width="1330" height="30" rx="8" fill="#764ba2" opacity="0.8"/>
                <text x="675" y="470" text-anchor="middle" font-size="12" font-weight="bold" fill="white">System Architecture & Implementation</text>
                
                <!-- Scalability -->
                <g transform="translate(30, 500)">
                    <rect x="0" y="0" width="300" height="180" rx="5" fill="#fff" stroke="#dee2e6"/>
                    <text x="150" y="20" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">Scalability Solutions</text>
                    
                    <text x="10" y="40" font-size="9" fill="#666">• Distributed Training:</text>
                    <rect x="10" y="45" width="280" height="20" rx="3" fill="#e3f2fd"/>
                    <text x="15" y="58" font-size="8" fill="#333">AWS SageMaker with multi-GPU (p3.8xlarge)</text>
                    
                    <text x="10" y="80" font-size="9" fill="#666">• Graph Partitioning:</text>
                    <rect x="10" y="85" width="280" height="20" rx="3" fill="#e8f5e9"/>
                    <text x="15" y="98" font-size="8" fill="#333">Metis partitioning, 1B+ nodes across 32 partitions</text>
                    
                    <text x="10" y="120" font-size="9" fill="#666">• Incremental Learning:</text>
                    <rect x="10" y="125" width="280" height="20" rx="3" fill="#fff3e0"/>
                    <text x="15" y="138" font-size="8" fill="#333">Mini-batch updates without full retraining</text>
                    
                    <text x="10" y="160" font-size="9" fill="#666">• Throughput: 100K events/sec on 8 GPUs</text>
                </g>
                
                <!-- Evaluation Metrics -->
                <g transform="translate(350, 500)">
                    <rect x="0" y="0" width="300" height="180" rx="5" fill="#fff" stroke="#dee2e6"/>
                    <text x="150" y="20" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">Evaluation Benchmarks</text>
                    
                    <text x="10" y="40" font-size="9" font-weight="bold" fill="#666">Synthetic Dataset:</text>
                    <text x="10" y="55" font-size="8" fill="#333">• 10M nodes, 100M edges</text>
                    <text x="10" y="68" font-size="8" fill="#333">• 15 attack types injected</text>
                    <text x="10" y="81" font-size="8" fill="#333">• Varying complexity levels</text>
                    
                    <text x="10" y="100" font-size="9" font-weight="bold" fill="#666">Real-world Data (Amazon):</text>
                    <text x="10" y="115" font-size="8" fill="#333">• Production logs (anonymized)</text>
                    <text x="10" y="128" font-size="8" fill="#333">• Actual security incidents</text>
                    <text x="10" y="141" font-size="8" fill="#333">• Cross-validation with SOC alerts</text>
                    
                    <text x="10" y="160" font-size="9" font-weight="bold" fill="#666">Target Metrics:</text>
                    <text x="10" y="173" font-size="8" fill="#333">Precision: >0.95 | Recall: >0.90 | F1: >0.92</text>
                </g>
                
                <!-- Attack Types -->
                <g transform="translate(670, 500)">
                    <rect x="0" y="0" width="320" height="180" rx="5" fill="#fff" stroke="#dee2e6"/>
                    <text x="160" y="20" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">Detectable Attack Patterns</text>
                    
                    <rect x="10" y="30" width="145" height="35" rx="3" fill="#ffebee"/>
                    <text x="82" y="45" text-anchor="middle" font-size="8" font-weight="bold" fill="#d32f2f">Data Exfiltration</text>
                    <text x="82" y="58" text-anchor="middle" font-size="7" fill="#666">Multi-hop access chains</text>
                    
                    <rect x="165" y="30" width="145" height="35" rx="3" fill="#fce4ec"/>
                    <text x="237" y="45" text-anchor="middle" font-size="8" font-weight="bold" fill="#c2185b">Privilege Escalation</text>
                    <text x="237" y="58" text-anchor="middle" font-size="7" fill="#666">Agent delegation abuse</text>
                    
                    <rect x="10" y="70" width="145" height="35" rx="3" fill="#f3e5f5"/>
                    <text x="82" y="85" text-anchor="middle" font-size="8" font-weight="bold" fill="#7b1fa2">Account Takeover</text>
                    <text x="82" y="98" text-anchor="middle" font-size="7" fill="#666">Anomalous login patterns</text>
                    
                    <rect x="165" y="70" width="145" height="35" rx="3" fill="#ede7f6"/>
                    <text x="237" y="85" text-anchor="middle" font-size="8" font-weight="bold" fill="#512da8">API Abuse</text>
                    <text x="237" y="98" text-anchor="middle" font-size="7" fill="#666">Rate limit violations</text>
                    
                    <rect x="10" y="110" width="145" height="35" rx="3" fill="#e8eaf6"/>
                    <text x="82" y="125" text-anchor="middle" font-size="8" font-weight="bold" fill="#303f9f">Insider Threats</text>
                    <text x="82" y="138" text-anchor="middle" font-size="7" fill="#666">Unusual access patterns</text>
                    
                    <rect x="165" y="110" width="145" height="35" rx="3" fill="#e3f2fd"/>
                    <text x="237" y="125" text-anchor="middle" font-size="8" font-weight="bold" fill="#1976d2">Fraud Rings</text>
                    <text x="237" y="138" text-anchor="middle" font-size="7" fill="#666">Coordinated attacks</text>
                    
                    <text x="160" y="165" text-anchor="middle" font-size="8" fill="#666" font-style="italic">
                        Real-time detection with explainable alerts
                    </text>
                </g>
                
                <!-- Timeline & Budget -->
                <g transform="translate(1010, 500)">
                    <rect x="0" y="0" width="310" height="180" rx="5" fill="#fff" stroke="#dee2e6"/>
                    <text x="155" y="20" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">Project Timeline (12 months)</text>
                    
                    <rect x="10" y="30" width="290" height="25" rx="3" fill="#e8f5e9"/>
                    <text x="15" y="45" font-size="8" font-weight="bold" fill="#333">Months 1-3:</text>
                    <text x="80" y="45" font-size="8" fill="#666">Graph modeling & GNN architecture</text>
                    
                    <rect x="10" y="60" width="290" height="25" rx="3" fill="#fff3e0"/>
                    <text x="15" y="75" font-size="8" font-weight="bold" fill="#333">Months 4-6:</text>
                    <text x="80" y="75" font-size="8" fill="#666">Contrastive learning framework</text>
                    
                    <rect x="10" y="90" width="290" height="25" rx="3" fill="#e3f2fd"/>
                    <text x="15" y="105" font-size="8" font-weight="bold" fill="#333">Months 7-9:</text>
                    <text x="80" y="105" font-size="8" fill="#666">LLM integration & benchmarks</text>
                    
                    <rect x="10" y="120" width="290" height="25" rx="3" fill="#fce4ec"/>
                    <text x="15" y="135" font-size="8" font-weight="bold" fill="#333">Months 10-12:</text>
                    <text x="80" y="135" font-size="8" fill="#666">Evaluation & dissemination</text>
                    
                    <text x="10" y="160" font-size="9" font-weight="bold" fill="#333">Budget: $80K cash + $40K AWS credits</text>
                    <text x="10" y="173" font-size="8" fill="#666">50% student support | 35% PI | 15% dissemination</text>
                </g>
                
                <!-- Example Alert -->
                <rect x="30" y="700" width="1290" height="170" rx="8" fill="url(#alertGradient)"/>
                <text x="675" y="725" text-anchor="middle" font-size="12" font-weight="bold" fill="white">Example Real-time Security Alert</text>
                
                <rect x="50" y="740" width="1250" height="110" rx="5" fill="white" opacity="0.95"/>
                <text x="60" y="760" font-size="11" font-weight="bold" fill="#d32f2f">⚠️ CRITICAL SECURITY ALERT - Potential Data Exfiltration Detected</text>
                
                <text x="60" y="780" font-size="10" fill="#333" font-weight="bold">Timestamp:</text>
                <text x="150" y="780" font-size="10" fill="#666">2025-02-15 14:32:18 UTC</text>
                
                <text x="60" y="798" font-size="10" fill="#333" font-weight="bold">Pattern:</text>
                <text x="150" y="798" font-size="10" fill="#666">Unauthorized multi-hop data access through agent delegation chain</text>
                
                <text x="60" y="816" font-size="10" fill="#333" font-weight="bold">Attack Path:</text>
                <text x="150" y="816" font-size="10" fill="#666" font-family="monospace">User_8234 → Agent_Bot_17 → Internal_API_v2 → Customer_Database → S3_Bucket_Export</text>
                
                <text x="60" y="834" font-size="10" fill="#333" font-weight="bold">Risk Score:</text>
                <text x="150" y="834" font-size="10" fill="#d32f2f" font-weight="bold">0.94 (CRITICAL)</text>
                <text x="250" y="834" font-size="10" fill="#333">| Affected Records: ~50,000 | Action: Block & Investigate</text>
                
                <rect x="1100" y="775" width="180" height="40" rx="5" fill="#d32f2f"/>
                <text x="1190" y="800" text-anchor="middle" font-size="11" fill="white" font-weight="bold">IMMEDIATE ACTION</text>
            </g>
        </svg>
    </div>
</body>
</html>