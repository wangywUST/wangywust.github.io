# Recent Advances in Large Language Models: Reasoning, Agents, and Vision-Language Models

## Introduction  
Large Language Models (LLMs) have rapidly progressed from mere text predictors to versatile AI systems capable of complex reasoning, tool use, and multi-modal understanding. This presentation explores three major recent directions in LLM development:
1. **Reasoning LLMs** – techniques that enable step-by-step logical problem solving.
2. **Autonomous/Tool-Using Agents** – letting LLMs use external tools or act autonomously to complete tasks.
3. **Vision-Language Models (VLMs)** – combining visual processing with language understanding.

Each section delves into core concepts, examples (with inputs, intermediate reasoning, and outputs), comparative analyses, and notable research (papers & benchmarks like GSM8K, ARC, Toolformer, ReAct, MM1, GPT-4V). The goal is a deep conceptual understanding of how these advances make LLMs more powerful and general. We include tables, pseudocode, and illustrative figures (with placeholders) to clarify key ideas for a graduate-level audience familiar with transformer models and chat-based LLMs.

## 1. Reasoning in LLMs: From Answers to **Chain-of-Thought**  
Modern LLMs can do more than recite memorized facts – they can **reason** through complex tasks. *Reasoning LLMs* explicitly break down problems into intermediate steps before giving a final answer. This approach addresses the limitation of “one-shot” answering, especially for math, logic, or multi-step questions that standard LLM outputs often get wrong due to missing reasoning steps.

### 1.1 What Are “Reasoning LLMs”?  
A reasoning-enabled LLM is prompted or trained to **think step-by-step**, mimicking a human’s scratch work or internal monologue. Instead of producing an answer immediately, the model generates a **chain of thought (CoT)**: a sequence of intermediate reasoning steps that lead to the solution ([Chain-of-Thought Prompting](https://learnprompting.org/docs/intermediate/chain_of_thought?srsltid=AfmBOorqoS6ODSzmOmtU1sSNFK67aRe-sMPuUp-odJ7F2ZJLX3lb3B69#:~:text=A%3A)) ([Chain-of-Thought Prompting](https://learnprompting.org/docs/intermediate/chain_of_thought?srsltid=AfmBOorqoS6ODSzmOmtU1sSNFK67aRe-sMPuUp-odJ7F2ZJLX3lb3B69#:~:text=Incorrect%20Solution%20)). These steps can be thought of as the model’s “intermediate scratchpad” where it works through the problem before concluding. By making reasoning explicit, we get two benefits: 
- **Better Accuracy** on complex problems (the model is less likely to skip logic).
- **Interpretability**, as we can inspect the reasoning the model followed.

**Chain-of-Thought Prompting:** Introduced by Wei et al. (2022), CoT prompting involves giving the model examples where the reasoning process is written out. This cues the model to follow suit ([Chain-of-Thought Prompting](https://learnprompting.org/docs/intermediate/chain_of_thought?srsltid=AfmBOorqoS6ODSzmOmtU1sSNFK67aRe-sMPuUp-odJ7F2ZJLX3lb3B69#:~:text=The%20example%20below%20illustrates%20the,more%20accurate%20and%20interpretable%20outcomes)) ([Chain-of-Thought Prompting](https://learnprompting.org/docs/intermediate/chain_of_thought?srsltid=AfmBOorqoS6ODSzmOmtU1sSNFK67aRe-sMPuUp-odJ7F2ZJLX3lb3B69#:~:text=Applications%20and%20Benefits%3A)). Even without further training, simply adding *“Let’s think step by step”* or showing worked solutions in the prompt can elicit multi-step reasoning from a sufficiently large model.

**Example – Direct vs. Chain-of-Thought:**  
Consider a math word problem: 

*Question:* *“If Alice has 5 apples and buys 7 more, then gives 3 to Bob, how many apples does Alice have?”*  

- **Standard LLM (direct answer):** *“Alice has 9 apples.”* (The model might do this in one step mentally: 5+7–3 = 9.)  

- **LLM with Chain-of-Thought:**  
  *Thought 1:* “Alice starts with 5 apples and buys 7, so now she has 5+7 = 12.”  
  *Thought 2:* “She then gives away 3, so 12–3 = 9.”  
  *Answer:* “$\displaystyle 9$.”  

Here the chain-of-thought makes the calculation explicit. For simple arithmetic both approaches got it right, but on harder problems the direct method often fails whereas the CoT method succeeds by breaking the task into subtasks.

### 1.2 Why Chain-of-Thought Helps  
Reasoning in steps allows the model to tackle **multi-step logic, arithmetic, or commonsense** increments rather than leaping to an answer. This significantly improves performance on challenging benchmarks:
- **GSM8K (Grade School Math)** – a dataset of math word problems. Prompting a 540B model (PaLM) with CoT boosted solve accuracy from *17.9% (standard)* to **58.1%** – a state-of-the-art result at the time ([A Comprehensive Guide to Chain-of-Thought Prompting - Future Skills Academy](https://futureskillsacademy.com/blog/chain-of-thought-prompting/#:~:text=You%20can%20notice%20examples%20of,tuning%20or%20special%20training)). In other words, with CoT the model solved over 3× more problems correctly than with a direct approach.
- **ARC-Challenge (AI2 Reasoning Challenge)** – a hard science question dataset. CoT and related strategies greatly improved performance on this and similar logic benchmarks, approaching or surpassing average human scores as model size grew. (For instance, GPT-4 scored around the 80% range on ARC, nearing human-level, thanks in part to enhanced reasoning ability.)
- Other tasks like **MATH (math competition problems)**, **CSQA (commonsense QA)**, and symbolic reasoning puzzles also saw substantial gains ([Chain-of-Thought Prompting](https://learnprompting.org/docs/intermediate/chain_of_thought?srsltid=AfmBOorqoS6ODSzmOmtU1sSNFK67aRe-sMPuUp-odJ7F2ZJLX3lb3B69#:~:text=Task%20Model%20Standard%20Prompting%20Accuracy,35)) ([Chain-of-Thought Prompting](https://learnprompting.org/docs/intermediate/chain_of_thought?srsltid=AfmBOorqoS6ODSzmOmtU1sSNFK67aRe-sMPuUp-odJ7F2ZJLX3lb3B69#:~:text=thought%2C%20which%20led%20to%20worse,the%20size%20of%20the%20model)). The table below shows how CoT prompting dramatically boosts accuracy across various tasks for a large model:

| **Benchmark Task**       | **Standard Prompt Accuracy** | **CoT Prompt Accuracy** | **Improvement**       |
|--------------------------|------------------------------|-------------------------|-----------------------|
| GSM8K (Math word problems)     | 17.9% ([A Comprehensive Guide to Chain-of-Thought Prompting - Future Skills Academy](https://futureskillsacademy.com/blog/chain-of-thought-prompting/#:~:text=You%20can%20notice%20examples%20of,tuning%20or%20special%20training)) (PaLM 540B)    | 58.1% ([A Comprehensive Guide to Chain-of-Thought Prompting - Future Skills Academy](https://futureskillsacademy.com/blog/chain-of-thought-prompting/#:~:text=You%20can%20notice%20examples%20of,tuning%20or%20special%20training)) (PaLM 540B)   | **+40.2%**   |
| ARC-Challenge (Science QA)    | ~70% (GPT-3.5)            | ~80% (GPT-4 w/ CoT)     | **+10%** (approx.) |
| MATH (Competition problems)   | low (GPT-3)               | high (GPT-4 + CoT)      | big increase (GPT-4 solves many problems) |
| Commonsense QA (CSQA)    | 76% (PaLM) ([Chain-of-Thought Prompting](https://learnprompting.org/docs/intermediate/chain_of_thought?srsltid=AfmBOorqoS6ODSzmOmtU1sSNFK67aRe-sMPuUp-odJ7F2ZJLX3lb3B69#:~:text=Task%20Model%20Standard%20Prompting%20Accuracy,35))    | 80% (PaLM + CoT) ([Chain-of-Thought Prompting](https://learnprompting.org/docs/intermediate/chain_of_thought?srsltid=AfmBOorqoS6ODSzmOmtU1sSNFK67aRe-sMPuUp-odJ7F2ZJLX3lb3B69#:~:text=Task%20Model%20Standard%20Prompting%20Accuracy,35))   | +4%         |
| Symbolic Reasoning       | ~60% (PaLM) ([Chain-of-Thought Prompting](https://learnprompting.org/docs/intermediate/chain_of_thought?srsltid=AfmBOorqoS6ODSzmOmtU1sSNFK67aRe-sMPuUp-odJ7F2ZJLX3lb3B69#:~:text=Task%20Model%20Standard%20Prompting%20Accuracy,35))   | ~95% (PaLM + CoT) ([Chain-of-Thought Prompting](https://learnprompting.org/docs/intermediate/chain_of_thought?srsltid=AfmBOorqoS6ODSzmOmtU1sSNFK67aRe-sMPuUp-odJ7F2ZJLX3lb3B69#:~:text=SVAMP%20%28Math%29PaLM%20540B%2057,35))  | **+35%**     |

*Table:* **Effect of Chain-of-Thought (CoT) Reasoning on Performance.** CoT prompts substantially improve accuracy, especially for complex tasks, when used with large models (100B+ parameters) ([Chain-of-Thought Prompting](https://learnprompting.org/docs/intermediate/chain_of_thought?srsltid=AfmBOorqoS6ODSzmOmtU1sSNFK67aRe-sMPuUp-odJ7F2ZJLX3lb3B69#:~:text=Importantly%2C%20according%20to%20CoT%20authors%2C,the%20size%20of%20the%20model)). Smaller models (<10B) often cannot follow CoT correctly, but big models leverage it to reason effectively ([Chain-of-Thought Prompting](https://learnprompting.org/docs/intermediate/chain_of_thought?srsltid=AfmBOorqoS6ODSzmOmtU1sSNFK67aRe-sMPuUp-odJ7F2ZJLX3lb3B69#:~:text=Importantly%2C%20according%20to%20CoT%20authors%2C,the%20size%20of%20the%20model)).

The improvements show that prompting the model to “think out loud” mitigates errors from trying to do too much in one step. It also reduces hallucination in reasoning since each step can be checked against the problem.

![Image: Chain-of-Thought vs Standard Prompting]()  
*Illustration:* **Step-by-step CoT vs. direct answer.** The left side shows a naive single-step answer (often incorrect for hard problems), while the right side depicts an LLM enumerating reasoning steps, leading to a correct, justified answer. (By writing out the logic, the model reaches the correct conclusion more reliably.)

### 1.3 Advanced Reasoning Techniques  
**Few-Shot vs. Zero-Shot CoT:** The initial CoT work used few-shot prompting (providing example solutions). Later, a *Zero-Shot CoT* method was found: simply appending a trigger phrase like *“Let’s think step by step”* to the user’s question often induces the model to produce a chain-of-thought even without explicit examples. This works surprisingly well for GPT-3.5/4 class models on many tasks, essentially telling the model to employ CoT reasoning on the fly.

**Self-Consistency:** One challenge with CoT is that the generated reasoning might occasionally go astray. *Self-consistency* (Wang et al. 2022) is a technique where the LLM is prompted to generate multiple independent chains-of-thought and answers, then the final answer is chosen by a majority vote or confidence measure across these attempts. This reduces the chance of accepting a flawed single chain-of-thought. It leverages the idea that while any one chain might have an error, the most common answer across many reasoning paths is likely correct. This yielded further performance boosts on GSM8K and other benchmarks beyond a single CoT run.

**Tools and External Checks:** (Transitioning to next section) Even with step-by-step reasoning, LLMs can struggle with tasks like exact arithmetic or up-to-date factual questions. An emerging idea is to let the model call external tools *during* its reasoning (e.g. a calculator for arithmetic, or a search engine for facts). This blurs into the territory of *LLM agents* (Section 2). For example, a reasoning LLM might decide “I should use a calculator for this step” to avoid arithmetic mistakes. This combination of internal reasoning and external tool use is at the frontier of LLM reasoning research (e.g., the ReAct framework discussed soon).

**Beyond Linear Chains – Tree-of-Thought:** Recent research explores letting the model consider a *tree* of possible reasoning steps (branching out different approaches) and then search or evaluate the best branch (analogous to game tree search). This “Tree-of-Thought” approach is experimental but aims to handle especially tricky problems by not committing to one line of thought too early. It’s like brainstorming multiple solution paths and then converging on the most promising one. Early results show it can solve some problems that linear CoT cannot, by recovering from false starts on a different branch.

### 1.4 Example: Chain-of-Thought in Action  
Let’s walk through a concrete example demonstrating chain-of-thought reasoning on a more complex question. Consider a typical **GSM8K** problem:

**Question:** *“Jenny’s age is twice Michael’s age. Five years ago, Jenny was three times as old as Michael. How old is Jenny now?”*

A standard LLM might confuse the relations or make a guess. A reasoning LLM will break it down:

1. *Let’s think step by step.* We have Jenny’s age = 2 × Michael’s age (currently). Let J and M be their current ages.
2. Equation 1: $J = 2M$.
3. Five years ago, Jenny’s age was $J - 5$, Michael’s was $M - 5$.
4. At that time, “Jenny was three times as old as Michael” gives Equation 2: $J - 5 = 3(M - 5)$.
5. Substitute $J = 2M$ into Eq2: $2M - 5 = 3(M - 5)$.
6. Solve: $2M - 5 = 3M - 15$ $\implies$ bring terms: $-5 + 15 = 3M - 2M$ $\implies$ $10 = M$.
7. So Michael is 10, and $J = 2M = 20$.
8. *Answer:* Jenny is **20 years old**.

**Output (LLM):** “Jenny is 20 years old.”  

Here the model essentially did algebra by writing down the equations in English. Each step follows logically, and even a reader can follow how it reached the answer. This is the power of chain-of-thought prompting – the LLM not only gets the answer right, but shows the reasoning clearly.

### 1.5 Reasoning LLMs vs. Standard LLMs  
To summarize this section, we compare a vanilla LLM (treating it as a black box that directly maps input to output) and a reasoning-enabled LLM:

| **Aspect**            | **Standard LLM** (direct prompt)     | **Reasoning LLM** (CoT or similar)       |
|-----------------------|--------------------------------------|------------------------------------------|
| **Approach to questions** | Answers in one step by next-word prediction – no explicit intermediate output. | Generates a **chain of intermediate steps** (“thoughts”) before final answer. |
| **Interpretability**  | Low – the reasoning is internal and not visible. | High – the model’s thought process is shown step-by-step, aiding transparency. |
| **Performance on complex tasks** | Struggles with multi-step problems (math word problems, logical puzzles). Tends to make leaps or mistakes. | Excels at multi-step and logical tasks by tackling them stepwise ([A Comprehensive Guide to Chain-of-Thought Prompting - Future Skills Academy](https://futureskillsacademy.com/blog/chain-of-thought-prompting/#:~:text=You%20can%20notice%20examples%20of,tuning%20or%20special%20training)). Achieves higher accuracy on benchmarks (GSM8K, ARC, etc.) with CoT prompting. |
| **Error characteristics** | More likely to **hallucinate** reasoning or make arithmetic errors silently. | Can still make errors, but easier to **spot mistakes** in the chain. Allows techniques like self-consistency or manual review to correct steps. |
| **Model size needed** | Small models can answer factoid questions, but fail at complex reasoning. | CoT is most effective on **large models** (100B+ params) ([Chain-of-Thought Prompting](https://learnprompting.org/docs/intermediate/chain_of_thought?srsltid=AfmBOorqoS6ODSzmOmtU1sSNFK67aRe-sMPuUp-odJ7F2ZJLX3lb3B69#:~:text=Importantly%2C%20according%20to%20CoT%20authors%2C,the%20size%20of%20the%20model)) which have the capacity to follow logical prompts. Smaller models often produce incoherent chains. |
| **Example**           | Q: “What is 37×49?” → *“1800”* (hallucinated guess, no working shown) | Q: “What is 37×49?” → Thought: “37×50 =1850, subtract 37: 1850–37=1813.” Answer: “1813.” (shows calculation) |

In summary, enabling reasoning in LLMs via prompting or training is a **major advancement** that has made LLMs far more capable problem solvers. It laid the groundwork for further enhancements – including the ability to use **external tools** when reasoning, which we discuss next.

## 2. Autonomous and Tool-Using Agents  
While chain-of-thought lets an LLM reason internally, another leap is allowing LLMs to **take actions in the world**. An LLM *agent* can interact with external tools or environments (e.g. calling APIs, doing web searches, running code) in a loop of reasoning and acting. This makes LLMs *autonomous* to a degree – they can be given a goal and then **figure out how to fulfill it by themselves**, using tools along the way.

Why is this needed? Because even the best purely textual LLM has limitations: it has a fixed knowledge cutoff, it isn’t good at precise calculation or real-time data, and it cannot directly make changes in the world (like sending an email or executing code) just by outputting text. Tool use and autonomy address these gaps:
- **Tools extend LLM capabilities**: e.g. a calculator for math, a search engine for up-to-date info, a database or code interpreter, etc.
- **Autonomy (multi-step planning)** allows the model to break a complex goal into sub-tasks, pursue each sub-task (possibly with tools), and adjust if needed – rather than relying on a human to prompt for every intermediate step.

### 2.1 LLMs as Agents: What Does It Mean?  
An LLM agent typically follows a loop: **(Observe environment ⇒ Reason ⇒ Act ⇒ Observe new info ⇒ …)** until a task is done. The “environment” could be tools like web search or even a simulated world. Unlike a single-turn Q&A, the LLM agent engages in an interactive process.

**Key Components of LLM Agents:**  
- **Observation**: the agent sees the current state (e.g. user query, or results from last action).
- **Reasoning (Thought)**: the LLM decides what to do next. This is often captured as a textual *thought* (e.g. “I should look up who this person is.”).
- **Action**: the LLM outputs an *action command* instead of a final answer. For example, it might output something like `Search["Apple Remote original program"]`. The system executing the agent sees this and performs the action (calls a search API).
- **Observation (Result)**: The result of the action (search results text) is fed back to the LLM.
- The cycle repeats: the LLM incorporates the new information, reasons again, possibly takes another action, and so on. Eventually, it outputs a final **answer or solution** when done.

This architecture lets the LLM **branch out of its own internal knowledge** and use external information or capabilities as needed.

### 2.2 Tool Use: From Plugins to Toolformer  
OpenAI’s ChatGPT introduced **plugins** in 2023 which essentially turn it into an agent: the model can decide to call a plugin (tool) like a web browser, calculator, or booking service. *“One of the newest and most underrated upgrades to ChatGPT is the plugin feature – the LLM can now decide on its own to use tools to perform actions outside of simple text responses, like booking a flight or fact-checking itself”* ([ Toolformer: Giving Large Language Models… Tools  | by Boris Meinardus | Medium](https://medium.com/@boris.meinardus/toolformer-giving-large-language-models-tools-1562c3bf69fb#:~:text=One%20of%20the%20newest%20and,checking%20itself)). This was a big practical leap: suddenly LLMs could retrieve real up-to-date information, do computations, or interact with third-party services.

**Toolformer (2023)** – a research project by Meta – took this idea further by training the model itself to insert API calls into its generation ([[2302.04761] Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761#:~:text=achieve%20the%20best%20of%20both,with%20much%20larger%20models%2C%20without)). The model was taught (in a self-supervised way) to decide *when* a tool could help and to output a call like `[Calculator(432 * 19) -> 8208]` mid-sentence, get the result, and use it in the continuation. Remarkably, Toolformer (based on a 6.7B model) **achieved substantially improved zero-shot performance on various tasks by using tools, often matching much larger (untuned) models** ([[2302.04761] Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761#:~:text=achieve%20the%20best%20of%20both,with%20much%20larger%20models%2C%20without)). In other words, a medium-sized LLM with tool-use abilities can out-perform a much bigger LLM that’s stuck with its internal knowledge. Tools give “superpowers” without needing to scale the model as much.

Notable tools for LLMs include:
- **Calculator** – for arithmetic and math (LLMs often make mistakes in math, so delegating to a calculator yields exact results).
- **Search engine / Wikipedia** – for up-to-date facts or detailed info on obscure queries.
- **Database or QA system** – some systems use a vector database to find relevant context (related to Retrieval-Augmented Generation, a separate but related idea).
- **Code execution** – e.g. Python interpreter: the LLM can write code to compute an answer or simulate something (this approach is used in OpenAI’s “Code Interpreter” tool).
- **Translator** – an LLM might call an external translation API if needed (though modern LLMs themselves are good at translation).
- **Custom APIs** – e.g. scheduling a meeting, controlling a robot, etc. The sky’s the limit if the model knows the API.

**Toolformer’s Approach:** It provided a handful of examples of how to use each API, then let the model practice on unlabeled text, figuring out where an API call would help predict the next token better ([[2302.04761] Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761#:~:text=factual%20lookup%2C%20where%20much%20simpler,shot%20performance%20across%20a)). Through this, it “taught itself” where using a tool makes sense. For instance, in text about dates it might learn to call a date calculation API instead of guessing the date difference. By fine-tuning on this augmented data, the model learned to seamlessly intermix API calls with natural language. 

This was a **training-time** augmentation. Alternatively, one can do it at **inference-time** via prompting – that’s where frameworks like ReAct come in.

### 2.3 ReAct: Reasoning + Acting (in Prompt)  
**ReAct** (Yao et al. 2022) is a framework that *combines chain-of-thought reasoning with actions* in a single prompting paradigm ([ReAct Prompting | Prompt Engineering Guide<!-- --> ](https://www.promptingguide.ai/techniques/react#:~:text=ReAct%20is%20a%20general%20paradigm,involved%20to%20perform%20question%20answering)). Instead of just prompting the model for reasoning steps, we also prompt it with an **action format**. A ReAct prompt typically includes few-shot examples of an agent solving tasks, with a transcript like:

```
Thought: I need to find more information about X  
Action: Search("X")  
Observation: [result of search]  
Thought: The result suggests Y...  
Action: Lookup("Y detail")  
Observation: ...  
Thought: Now I have enough info to answer.  
Answer: [final answer here]
```

The model, seeing this format, will generate both “Thought” and “Action” lines. The key is that we interleave them: the model produces a thought (reasoning) which leads to an action, gets new info, reasons further, and so on. ReAct thus **synergizes reasoning and acting** ([ReAct Prompting | Prompt Engineering Guide<!-- --> ](https://www.promptingguide.ai/techniques/react#:~:text=ReAct%20is%20a%20general%20paradigm,involved%20to%20perform%20question%20answering)). The reasoning trace helps the model decide the next action, and the retrieved information informs the subsequent reasoning – a positive feedback loop.

**Benefits:** ReAct was shown to **outperform prior baselines** on knowledge-intensive tasks (like open-domain QA) and decision-making tasks ([ReAct Prompting | Prompt Engineering Guide<!-- --> ](https://www.promptingguide.ai/techniques/react#:~:text=Results%20show%20that%20ReAct%20can,external%20information%20obtained%20during%20reasoning)). By retrieving relevant facts in the middle of its reasoning, it greatly reduces hallucinations and errors. It also makes the process **interpretable and controllable** – you can watch the agent’s chain-of-thought and intervene if needed. In fact, *“ReAct leads to improved human interpretability and trustworthiness of LLMs”* and the best results were achieved when combining ReAct with chain-of-thought prompting ([ReAct Prompting | Prompt Engineering Guide<!-- --> ](https://www.promptingguide.ai/techniques/react#:~:text=Results%20show%20that%20ReAct%20can,external%20information%20obtained%20during%20reasoning)) – essentially using CoT-style thinking for planning actions, which allows use of both internal knowledge and external information.

![Image: ReAct Agent Loop]()  
*Illustration:* **ReAct agent reasoning and acting.** The LLM iteratively generates a Thought (blue) explaining what it will do, then an Action (green) which is executed, then sees an Observation (yellow) from the environment. This loop continues until the LLM produces a final answer. Such prompting lets the model handle complex queries by gathering information as needed, rather than relying only on built-in knowledge.

**Example – ReAct in practice:**  
*User query:* “Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?” (This is a question requiring multi-hop reasoning: identify what “program Apple Remote was designed to interact with”, then find what other devices can control that program.)  

A ReAct-enabled agent might proceed: 

- **Thought 1:** “The question asks: Apple Remote (the device) was originally designed to interact with some program. Which program? And then: what other devices can control that program? I should find the program first.”  
- **Action 1:** `Search("Apple Remote originally designed to interact with")`  
- **Observation 1:** *... search results show: "Apple Remote ... originally designed to interact with iTunes" ...*  
- **Thought 2:** “It’s iTunes. Now the question: what other devices can control iTunes?”  
- **Action 2:** `Search("devices that can control iTunes")`  
- **Observation 2:** *... results: "iTunes can be controlled by Apple Remote, and also by iPhone, iPad via Remote app, etc."*  
- **Thought 3:** “It seems other devices include iPhones, iPads (via the Remote app), and possibly third-party remotes.”  
- **Answer:** “It was designed for iTunes, which can also be controlled by devices like the iPhone or iPad (running the Remote app) in addition to the Apple Remote.”  

This illustrates how the agent figured out the answer via two web searches, something a single-turn LLM without tool use might not have known. The *thoughts* guided the search actions, and the retrieved info was integrated into the reasoning. ReAct prompting enabled this entire chain inside the LLM.

**Pseudocode: ReAct Agent Loop** (simplified):  
```python
state = initial_question
while True:
    output = LLM(prompt_with(state))  
    # LLM generates either a Thought, an Action, or Final Answer based on prompt format.
    if output.type == "Action":
        result = execute_tool(output)
        state += "\nObservation: " + result  # add result to the prompt
        continue  # loop back for another thought
    elif output.type == "Answer":
        print("Final Answer:", output.text)
        break
```
This loop continues until the model emits an answer rather than an action. In prompt engineering terms, the prompt contains the dialogue of thoughts/actions, and each iteration extends it. This is how frameworks like LangChain implement LLM agents using ReAct – by programmatically detecting the “Action:” and feeding back the tool’s result.

### 2.4 Autonomous Agents: Beyond Single Tools  
With the ability to use tools, developers combined it with **goal-driven loops** to create autonomous agents like **AutoGPT** and **BabyAGI** (popular open-source projects in 2023). These tie an LLM to a cycle of:
- Taking a high-level goal (e.g. “Research and write a report on XYZ”),
- Breaking it into sub-tasks,
- Executing tasks (using tools or the LLM itself for each),
- Generating new tasks from results until the goal is completed.

These systems often maintain a *task list* and a memory, allowing the LLM to keep track of progress. For example, AutoGPT can spawn new “thoughts” like “I should search for information A, then use that to get B, then compose a report.” It then carries out the plan with minimal human intervention, effectively acting like an **autonomous agent** that iteratively prompts itself.

**HuggingGPT** (Microsoft, 2023) demonstrated an agent that uses an LLM (ChatGPT) as a controller to orchestrate multiple AI models on Hugging Face for complex tasks (e.g., a multi-step task involving image generation, object detection, and language). The LLM decides which specialized model to call at each step – a form of tool use where tools are other AI models.

**Generative Agents (Interactive Sims)** (Stanford, 2023) took autonomy in a different direction – they put multiple LLM-based agents in a simulated game environment (like The Sims) to see if they could exhibit believable, emergent behaviors. Each agent could make plans (e.g. “go to the cafe at 3pm to meet a friend”) and remember interactions. This showcases that when given long-term memory and goals, LLM agents can indeed act in an autonomous, adaptive manner over extended periods, not just single Q&A sessions.

### 2.5 Comparison: Agent vs. Plain LLM Prompting  
It’s important to understand how this new agent paradigm contrasts with the classic single-turn prompt usage:

| **Characteristic**       | **Plain LLM Prompt**           | **LLM as Agent**                |
|--------------------------|-------------------------------|---------------------------------|
| **Interaction Style**    | One-shot or few-shot query → response. No follow-up by the model; any iteration is driven by the user. | Multi-turn *loop*. The LLM can *initiate* actions and request information. It’s an interactive **dialog** between the LLM and tools. |
| **Use of External Info** | Limited to what’s in model’s training data or provided in prompt. Cannot fetch new data mid-response. | **Can call tools/APIs** to get fresh info (web search, DB queries, etc.) ([ Toolformer: Giving Large Language Models… Tools  | by Boris Meinardus | Medium](https://medium.com/@boris.meinardus/toolformer-giving-large-language-models-tools-1562c3bf69fb#:~:text=One%20of%20the%20newest%20and,checking%20itself)). Can incorporate real-time data and computation results into its reasoning. |
| **Problem Solving**      | Solves in one step. Struggles with lengthy or decomposed tasks unless user manually breaks it down. | **Can decompose tasks** itself. Handles more complex goals by planning sub-tasks, executing them sequentially. More **autonomous** in figuring out what to do next. |
| **Memory**               | Limited to prompt window per turn (though can have some long context, it’s passive). | Can implement **long-term memory** via storage (e.g., the agent can save notes or update a context that persists across turns). More like a cognitive loop than a one-off response. |
| **Transparency**         | Only final answer is seen (unless model is prompted to explain). Harder to diagnose errors. | Intermediate **thoughts and actions are visible** (by design in ReAct). Easier to trace how it got to an answer; one can debug which action led to an error. |
| **Examples**             | Q: “What’s the capital of France?” → *“Paris.”* (No external call, answer from knowledge) | Q: “Who won the Best Actor Oscar in 2020 and give one of their movie quotes.” → Agent might Search for Oscar 2020 Best Actor (finds Joaquin Phoenix), then search for famous quotes by him, then respond with the info. |

In essence, agentic LLMs are **more powerful and flexible** – they *decide how to solve* a problem, rather than just solving it in one shot. However, this comes with challenges:
- The agent might get caught in loops or take irrelevant actions if not properly constrained.
- There’s higher complexity in orchestrating the prompt format, tool APIs, and maintaining state.
- Cost can be higher (multiple API calls to the LLM and tools).
- Ensuring **safety** is trickier: an autonomous agent could potentially do harmful things if instructed maliciously (e.g. use a tool to send spam emails). Safeguards and monitored execution are needed.

### 2.6 Notable Research and Developments in LLM Agents  
- **ReAct (2022)** – Already discussed; a seminal approach combining reasoning and acting in prompting ([ReAct Prompting | Prompt Engineering Guide<!-- --> ](https://www.promptingguide.ai/techniques/react#:~:text=ReAct%20is%20a%20general%20paradigm,involved%20to%20perform%20question%20answering)). It influenced many tool-using agent frameworks (LangChain’s agents are based on ReAct format, for example).
- **MRKL (2022)** – An earlier concept (Modular Reasoning, Knowledge and Language) that routed an LLM’s queries to different tools or experts. It was a precursor to the idea of an LLM orchestrating tool use.
- **Toolformer (2023)** – Fine-tuned model that learned tool API usage self-supervised ([[2302.04761] Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761#:~:text=achieve%20the%20best%20of%20both,with%20much%20larger%20models%2C%20without)). Showed even relatively small models gain a lot by using tools (often matching much larger models that lack tool-use).
- **HuggingGPT (2023)** – The LLM as a master controller calling other models for specific tasks (e.g., using a vision model for an image task, a speech model for audio, etc.). It treats each model as a tool and sequences calls to them based on the high-level request.
- **AutoGPT/BabyAGI (2023)** – Community-driven agent examples that popularized the concept of an “AI agent” that can iteratively improve and work towards open-ended goals. They showed the excitement (and pitfalls) of letting GPT-4 run autonomously (users found they can be creative but sometimes hilariously inept or stuck).
- **Self-Refine / Reflexion** – Methods where the LLM agent can critique its own outputs or mistakes and try again (essentially giving it a reflective capability to avoid repeating errors).
- **Real-world agents**: Efforts to use LLM agents in physical domains, like robotics (e.g., an LLM controlling a robot with actions like “pick up the cube” – possibly using a tool API that interfaces with robot controls). Also, agents in simulated environments (game AI, as mentioned with Generative Agents).

The agent paradigm is pushing us toward more **interactive AI**. Instead of just answering questions, LLMs are starting to function as **cognitive engines** that can *do things*: read the web, manipulate files, control other applications, etc. This opens up many possibilities – an AI that can research a topic thoroughly and then write a summary, or an AI that can take a user’s request “Plan my weekend trip” and actually go book hotels, find restaurants (by using tools).

It also raises new research questions on how to ensure these agents remain reliable, safe, and efficient. Combining reasoning with action is a big step toward more general AI behavior.

**Key Takeaway:** *Autonomous and tool-using agents extend LLMs beyond text prediction – they can interact with external systems and iteratively plan, making them far more capable on complex, real-world tasks than static prompts. This is a major frontier in 2024–2025 LLM research and applications.* The next section will look at another frontier: extending LLMs to **multimodal inputs, especially vision**, which further broadens what these models can do.

## 3. Vision-Language Models (VLMs): Multimodal LLMs  
Humans don’t only communicate with text – we perceive the world in images, sounds, etc. A long-standing goal in AI has been to build models that can **see and talk**: interpret images and describe them, or take visual context into account for reasoning. Vision-Language Models (VLMs) are systems that integrate visual processing with language understanding/generation. Recent advances have effectively combined the power of LLMs with powerful vision models, creating **multimodal LLMs** that accept images (and sometimes other modalities like audio) as part of their input and produce useful language outputs (captions, answers, explanations, etc.).

Why is this important? Consider tasks like:
- Describing what’s happening in a picture (image captioning).
- Answering a question about an image (Visual QA): e.g. “What is the color of the car in this image?”.
- Explaining a meme or infographic.
- Interpreting a chart or graph and answering questions about it.
- Helping a visually impaired user by understanding an image they send.
- Multimodal reasoning: e.g. “From the plot in this paper, does increasing X appear to increase Y?” – requiring reading a graph and using knowledge.

A pure text model can’t do these, and a pure vision model doesn’t produce rich language. The synergy of VLMs enables such applications.

### 3.1 How Vision-Language Models Work (Conceptually)  
At a high level, a VLM combines a **visual encoder** (to process images into some representation) with a **language model** (to turn that representation into text, or to use it in reasoning). There are a few design patterns:

- **Dual Encoder (CLIP-style)**: One network for images, one for text, trained to produce embeddings that match for the same concept. This was introduced by OpenAI’s **CLIP (Contrastive Language–Image Pre-training)** in 2021 ([CLIP: Connecting text and images | OpenAI](https://openai.com/index/clip/#:~:text=We%E2%80%99re%20introducing%20a%20neural%20network,shot%E2%80%9D%20capabilities%20of%20GPT%E2%80%912%20and%C2%A0GPT%E2%80%913)). CLIP learned visual concepts from 400M image-text pairs by training such that, say, the embedding of an image of a dog is close to the embedding of the text “a dog”. CLIP itself isn’t a generator, but it’s great at understanding images and mapping them to labels or captions via similarity. It demonstrated zero-shot image classification: provide the names of classes (text) and it will pick which is most similar to the image embedding, effectively classifying without explicit training on those classes ([CLIP: Connecting text and images | OpenAI](https://openai.com/index/clip/#:~:text=We%20present%20a%20neural%20network,image)). CLIP became a foundation for many later VLMs (it provides a strong vision encoder).

- **Encoder-Decoder (Captioning style)**: An image encoder produces features which are fed into a language decoder (like a GPT or a Transformer decoder) to generate a caption or answer. This is like the classic image captioning models, but scaled up with transformers. E.g., **ViLBERT, VisualBERT (2019)** fused image regions and text in a BERT-like model for vision-and-language understanding tasks. Those were early VLMs focused more on understanding than generation.

- **Interleaved multimodal models:** These models, like **Flamingo (DeepMind, 2022)**, accept sequences that mix images and text. Flamingo has a vision encoder (like a pre-trained EfficientNet or Vision Transformer) whose outputs are injected into a language model through special cross-attention layers ([Flamingo: a Visual Language Model for Few-Shot Learning - arXiv](https://arxiv.org/abs/2204.14198#:~:text=We%20introduce%20Flamingo%2C%20a%20family,bridge%20powerful%20pretrained)). It can take in a few example image–text pairs followed by a new image, and then generate text for the new image (few-shot learning). *“Flamingo… takes as input a prompt consisting of interleaved images, videos, and text and then outputs associated language,” achieving state-of-the-art few-shot learning on a wide range of open-ended multimodal tasks ([Tackling multiple tasks with a single visual language model - Google DeepMind](https://deepmind.google/discover/blog/tackling-multiple-tasks-with-a-single-visual-language-model/#:~:text=Today%2C%20in%20the%20preprint%20of,and%20then%20output%20associated%20language)).* This was a breakthrough – one model handling image+text in a prompted way, similar to how GPT-3 handles just text prompts. It set new SOTA on tasks like image captioning and visual question answering with only a handful of examples given (no task-specific finetuning) ([Tackling multiple tasks with a single visual language model - Google DeepMind](https://deepmind.google/discover/blog/tackling-multiple-tasks-with-a-single-visual-language-model/#:~:text=Today%2C%20in%20the%20preprint%20of,and%20then%20output%20associated%20language)).

- **Two-Stage Models (BLIP-2 style)**: An efficient approach used by **BLIP-2 (2023)** is to keep a **frozen** image encoder (like CLIP’s or ViT) and a frozen large language model (like GPT-style), and learn a small **bridge module** in between (the Q-Former). The Q-Former takes the image features and produces a handful of “query embeddings” that summarize the visual info, which are then fed into the LLM’s input embedding space ([BLIP-2: Bootstrapping Language-Image Pre-training with Frozen ...](https://arxiv.org/html/2301.12597#:~:text=the%20gap%20between%20a%20frozen,It%20extracts%20a%20fixed)). This way, the heavy vision and language components don’t need retraining – only the intermediate adapter is trained. BLIP-2 demonstrated you can connect a pre-trained ViT and a pre-trained LLM to get a capable VLM without full retraining. It can do zero-shot VQA and captioning quite well.

- **Multimodal Fine-tuned LLMs**: Models like **LLaVA, MiniGPT-4, and GPT-4’s vision** represent taking an LLM (sometimes a smaller fine-tuned one like Vicuna for MiniGPT-4, or the actual GPT-4 for OpenAI’s product) and feeding images into it via a projection. LLaVA (Large Language and Vision Assistant) in 2023 fine-tuned LLaMA (a language model) with images by mapping image features into the LLM’s embedding space and training on a conversational VQA dataset. The result is a chatbot that can “see” images and respond about them. OpenAI’s **GPT-4V** (GPT-4 Vision, launched 2023) is the most advanced example – it’s a version of GPT-4 that accepts image inputs in addition to text, and can output detailed answers or descriptions.

**Note:** Although initial VLM research had separate “vision encoders” and “text decoders”, the trend with GPT-4V and others is to unify them: essentially a giant model with both vision and language capabilities integrated. For instance, GPT-4’s vision component is not fully disclosed, but it likely uses a dedicated vision module whose output is fed into the same transformer as the text (some reports suggest a form of early fusion after the first few transformer layers). Apple’s **MM1 (2024)** also follows a unified large-scale multimodal model approach, training up to 30B parameter models that handle image+text together ([[2403.09611] MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611#:~:text=build%20MM1%2C%20a%20family%20of,thought%20prompting)).

### 3.2 Capabilities of Modern VLMs  
Today’s state-of-the-art VLMs, like GPT-4V and MM1, exhibit impressive capabilities:
- **Visual Description and Captioning:** They can generate detailed captions for images. E.g., show an image from a city street and the model can produce: “A busy street in New York City with yellow taxis in traffic and people crossing the crosswalk.”
- **Visual Question Answering (VQA):** You can ask a question about an image. For example, provide a photo and ask “How many people are in this photo and what are they doing?” The model can count the people and describe their activity.
- **OCR and Document Understanding:** GPT-4V can read text in images (OCR) and interpret it in context. E.g. you can upload a screenshot of a document and ask questions about it. It can also do things like analyze a chart image (bar graph, pie chart) and answer questions about the data.
- **Reasoning on images:** This is new – models like GPT-4V can do some reasoning that involves visual elements. For instance, explaining a joke in a meme image (which might require recognizing an object and understanding the cultural/contextual humor). One famous example: an image with a convoluted HDMI adapter chain was given to GPT-4 and it was asked “Why is this funny?” GPT-4V correctly explained that the absurdly impractical chain of adapters made it humorous. This kind of high-level reasoning on visual input was essentially unattainable before multimodal LLMs.
- **Multi-Image tasks:** Apple’s MM1 emphasizes *multi-image reasoning*, i.e., the model can take multiple images in the context (like frames of a comic or a before/after image) and reason across them ([[2403.09611] MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611#:~:text=training%20metrics%20and%20achieve%20competitive,thought%20prompting)) ([Apple’s MM1. Exploring the frontier of Multimodal… | by Simeon Emanuilov | Medium](https://medium.com/@simeon.emanuilov/apples-mm1-e7859ec252ac#:~:text=Enabling%20few,reasoning)). For example, comparing two medical images or tracking changes across time.
- **Enhanced few-shot learning:** MM1’s research found that training on interleaved image-text data gave their model strong *in-context learning* abilities with images (like Flamingo had). It can see a couple of examples of a new image task and then do a new one few-shot ([[2403.09611] MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611#:~:text=build%20MM1%2C%20a%20family%20of,thought%20prompting)).

**Notable Benchmarks:**  
Vision-language models are evaluated on an array of benchmarks:
- **COCO Captions** – generating captions for MS COCO images.
- **VQAv2** – Visual QA dataset where the model answers open-ended questions about images.
- **OK-VQA** – VQA requiring outside knowledge (not just what’s in the image, e.g., “What is the name of the monument in the picture?” requires knowing that monument’s name – a blend of vision and world knowledge).
- **NoCaps** – image captioning with novel objects not seen in training.
- **VizWiz** – VQA on images taken by blind users (often low-quality images, so a robust understanding needed).
- **TextCaps** – captioning images that contain text (requires reading the text in the image and incorporating it).
- **Science charts/diagrams** – e.g., the *MathVista* and *ChartQA* benchmarks test if models can interpret scientific plots or math diagrams.

Recent models have achieved impressive results:
- Apple’s **MM1-30B** **outperforms previous models (like Flamingo-80B by DeepMind, and Meta’s IDEFICS-80B) on captioning benchmarks (COCO, NoCaps, TextCaps) and on VizWiz-QA in few-shot settings ([Apple’s MM1. Exploring the frontier of Multimodal… | by Simeon Emanuilov | Medium](https://medium.com/@simeon.emanuilov/apples-mm1-e7859ec252ac#:~:text=Performance%20Apple%E2%80%99s%20MM1%20achieves%20impressive,shot%20settings)).** This is notable because it’s smaller than those models but better – thanks to a good training recipe. It shows how quickly the field is progressing that a 30B model (with MoE) can beat an 80B from a year prior on several tasks.
- However, the competition is fierce: the same report notes other models like **Emu-2 (37B)** and **LLaVA-Next (34B)** were competitive or better on VQAv2 and OK-VQA ([Apple’s MM1. Exploring the frontier of Multimodal… | by Simeon Emanuilov | Medium](https://medium.com/@simeon.emanuilov/apples-mm1-e7859ec252ac#:~:text=However%2C%20it%E2%80%99s%20important%20to%20note,the%20need%20for%20continued%20innovation)). So there’s no single model that dominates *every* task yet; some are better at factual QA, others at fine-grained captioning, etc. This indicates the field is still evolving and models are being specialized or tuned for certain strengths.
- **GPT-4V** (multimodal GPT-4) hasn’t been officially benchmarked on all academic datasets publicly, but anecdotal and limited evaluations suggest it’s at or near SOTA on many tasks. For example, one study found *GPT-4V outperformed other approaches on complex benchmarks like MathVista (diagram reasoning) and ChartQA (chart-based QA) ([Assessing GPT4-V on Structured Reasoning Tasks](https://arxiv.org/html/2312.11524v1#:~:text=Our%20results%20show%20that%20for,component%20actually%20results%20in%20lower))*. In VQAv2, early tests showed GPT-4V scored very highly (possibly even exceeding human performance in some settings), but those numbers are yet to be fully confirmed in publications.

**Table: Selected Vision-Language Models and Their Features**  

| **Model**        | **Year** | **Organization** | **Size**          | **Modalities**       | **Notable Capabilities**                                    |
|------------------|----------|------------------|-------------------|----------------------|-------------------------------------------------------------|
| **CLIP**         | 2021     | OpenAI           | ~400M (encoders)  | Image & Text (dual encoder) | Learned joint image-text embeddings; enabled zero-shot image classification matching ResNet-50 accuracy on ImageNet without using any labeled training images ([CLIP: Connecting text and images | OpenAI](https://openai.com/index/clip/#:~:text=We%20present%20a%20neural%20network,image)). Widely used as a vision backbone. |
| **Flamingo-80B** | 2022     | DeepMind         | 80B LLM + vision enc. | Image & Text (interleaved) | First large VLM for few-shot learning. Set SOTA on few-shot VQA, captioning ([Tackling multiple tasks with a single visual language model - Google DeepMind](https://deepmind.google/discover/blog/tackling-multiple-tasks-with-a-single-visual-language-model/#:~:text=Today%2C%20in%20the%20preprint%20of,and%20then%20output%20associated%20language)). Accepts interleaved image-text prompts (e.g. image, question about it, etc.). |
| **PaLI-17B**     | 2022     | Google           | 17B (encoder-decoder) | Image & Text (encoder-decoder) | Trained on multilingual image captions and VQA. Strong on captioning and VQA, handling multiple languages. Demonstrated multi-task learning (unifying vision tasks). |
| **BLIP-2**       | 2023     | Salesforce       | ~1B (ViT) + 6B (LLM) | Image & Text (two-stage) | Uses a frozen ViT and frozen LLM with a learned Q-Former bridge ([BLIP-2: Bootstrapping Language-Image Pre-training with Frozen ...](https://arxiv.org/html/2301.12597#:~:text=the%20gap%20between%20a%20frozen,It%20extracts%20a%20fixed)). Efficient yet effective zero-shot VQA and captioning. Can plug in different LLMs (e.g. Flan-T5 or GPT-like) for different trade-offs. |
| **Kosmos-1**     | 2023     | Microsoft        | 1.6B              | Image & Text (unified) | One of the first attempts at a multimodal GPT-like model. Showed ability on image-based IQ tests and joint vision-language tasks. Pioneered treating images as just another input token sequence in a transformer. |
| **LLaVA**        | 2023     | Community (UC Berkeley/MBZUAI) | 13B (LLaMA) | Image & Text (projected to LLM) | Fine-tuned an open LLM (LLaMA) with image inputs (via CLIP-ViT to LLM projection). Turns LLaMA into a multimodal chatbot that can do VQA and descriptions. Open-source analog of ChatGPT-vision (with lower capability). |
| **GPT-4V**       | 2023     | OpenAI           | ~>100B (est.)     | Image & Text (unified, multimodal) | GPT-4 with vision input. Outstanding reasoning on images: can interpret memes, diagrams, OCR text, solve visual puzzles. In a casual setting, can describe images or answer complex questions about them. Likely at or near SOTA on many vision-language tasks (OpenAI reported strong results on internal tests). |
| **Apple MM1-30B**| 2024     | Apple            | 30B (with MoE)    | Image & Text (unified) | Achieved SOTA on several benchmarks (captioning, VizWiz) outperforming larger models ([Apple’s MM1. Exploring the frontier of Multimodal… | by Simeon Emanuilov | Medium](https://medium.com/@simeon.emanuilov/apples-mm1-e7859ec252ac#:~:text=Performance%20Apple%E2%80%99s%20MM1%20achieves%20impressive,shot%20settings)). Supports **multi-image** inputs and few-shot chain-of-thought reasoning with images ([[2403.09611] MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611#:~:text=build%20MM1%2C%20a%20family%20of,thought%20prompting)) ([Apple’s MM1. Exploring the frontier of Multimodal… | by Simeon Emanuilov | Medium](https://medium.com/@simeon.emanuilov/apples-mm1-e7859ec252ac#:~:text=Enabling%20few,reasoning)). Signals Apple’s entry into large-scale VLMs. |

*(Table: A timeline of notable Vision-Language Models, showing the rapid progress in capabilities. Models have grown in sophistication from dual encoders like CLIP to fully integrated multimodal LLMs like GPT-4V and MM1.)*

### 3.3 Examples: Vision-Language Tasks  
To ground these concepts, let’s run through a few example scenarios a modern VLM can handle:

**Image Captioning Example:**  
*(Input: an image of a family having a picnic under a tree.)*  
**Model output:** “A family of four enjoys a picnic on a sunny day. A spread of food is laid out on a blanket under a large tree, while the parents and children smile and interact.”  
*Intermediate reasoning:* The model’s vision module identifies people, a picnic setup, the tree, sunny weather, etc., and the language module weaves it into a coherent sentence. (If we had the chain-of-thought, it might be something like: detects objects → infers scene → produces descriptive sentence.)

**Visual Question Answering Example:**  
*(Input: an image of a man standing next to a bicycle on a mountain trail, and the question: “Is the person likely on a leisure activity or working?”)*  
**Model reasoning (imagined):** Sees man with bicycle on a scenic trail, likely not in work uniform, context suggests recreation → “He is likely engaged in a leisure activity (biking on a trail for fun).”  
**Model answer:** “It appears to be a leisure activity – he’s biking on a mountain trail for recreation.”  
This requires understanding context and common sense from the image.

**OCR + Reasoning Example:**  
*(Input: a photo of a sign that reads “Parking $5 per hour, $20 per day”, question: “How much would 3 hours of parking cost?”)*  
The model must read the text “$5 per hour, $20 per day”, then do math.  
**Answer:** “3 hours would cost $15.”  
**Analysis:** The vision component does OCR (“$5 per hour, $20 per day”), the language/logic part interprets the pricing and calculates 3×$5. This blends vision (reading) with reasoning (simple math).

**Multi-image Reasoning Example:**  
Imagine feeding two images: one of a jar 3/4 full of water, another of the same jar 1/4 full, and asking: “Which photo shows more water and by what difference roughly?”  
The model needs to compare images: It can determine the first has more (3/4 vs 1/4, a difference of about half a jar).  
**Answer:** “The first image has significantly more water – roughly half a jar more water than the second image.”  
Some advanced models explicitly allow multiple image inputs in a single context (e.g., “Image1: ...; Image2: ...; Question: ...”).

**Explaining a Meme (Complex Reasoning) Example:**  
*(Input: an image meme – say a famous “Distracted Boyfriend” meme – with the question: “What is the joke here?”)*  
The model needs to identify the scene: A guy looking at another woman while his girlfriend is upset. Then recall or recognize this as a meme format about being distracted by something new.  
**Answer (GPT-4V style):** It would explain that the humor comes from the universal scenario of someone being distracted from their partner by someone else – often labeled in memes as e.g. “My study time” (girlfriend) and “a new video game” (the girl being looked at), etc. The model might say: “It’s the ‘Distracted Boyfriend’ meme, showing a man holding hands with his girlfriend but obviously turning to look at another woman. The joke is about **being distracted by something more appealing even when you should be paying attention to what you already have**, often used metaphorically.”  
This is high-level interpretation requiring not just object recognition but understanding the social context and the fact it’s a well-known internet meme format.

These examples show a spectrum from straightforward perception to sophisticated reasoning involving vision + language.

### 3.4 Challenges and Recent Research in VLMs  
Combining vision and language isn’t trivial. Some challenges and active research areas include:
- **Aligning the Modalities:** Training a model to jointly understand image and text without one dominating or confusing the other. (E.g., early attempts sometimes had the model just ignore the image and rely on priors to answer questions; careful loss balancing is needed to ensure image features are used.)
- **Efficiency:** Images are high-dimensional. Feeding an entire image as pixels into a transformer with text would be computationally huge. So we use CNN or ViT encoders to compress images first. Even then, handling multiple images or high-res images is heavy. Apple’s MM1 paper found that image resolution and token count significantly impact performance, whereas the exact architecture of the vision-language connector might be less critical ([[2403.09611] MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](https://arxiv.org/abs/2403.09611#:~:text=training%20data%20choices%2C%20we%20identified,to%2030B%20parameters%2C%20including%20both)). They also highlight the cost of multi-image input.
- **Visual hallucination:** Just like LLMs hallucinate facts, VLMs can hallucinate details not present in the image (e.g., saying “a red car” when the car was blue). Ensuring the model’s output stays grounded in the visual evidence is an ongoing concern. Some approaches to mitigate this include training with grounded learning objectives or using **visual prompting** (explicitly asking the model to describe what it sees before answering a question, to separate perception from prior knowledge).
- **Benchmarks beyond static images:** There’s interest in extending to **video** (temporal reasoning, e.g., describe a video or answer questions about a movie clip) and **audio** (truly multimodal models like Meta’s ImageBind aim to join image, text, audio, etc. in one space). GPT-4’s current public version doesn’t handle video or audio, but research models are tackling these.
- **In-context learning with vision:** Flamingo and MM1 show few-shot prompting can work (giving example image-question-answer triples). Others are exploring **instruction tuning** with multimodal data (like the open InstructBLIP which fine-tuned BLIP-2 on a variety of user instruction data with images).
- **Safety and ethics:** Vision adds new dimensions to consider, such as not revealing private info from images (face recognition, etc.), avoiding biases (not making problematic assumptions from appearance), and handling potentially graphic or sensitive images appropriately. OpenAI’s GPT-4V, for example, refuses to identify people in images and has filters for certain content. Research continues on making sure these models behave safely with visual input.

**Notable recent works and benchmarks:**
- **MMBench** or **MMBig**: large suites of tasks to evaluate multimodal LLMs on everything from simple captioning to complex reasoning and OCR.
- **Vision-augmented CoT (v-CoT)**: A method where you prompt the model to first describe the image and gather visual facts, then reason – essentially chain-of-thought but including visual descriptions. Early research suggests that prompting GPT-4V in this way can improve performance on tasks like chart reasoning ([Assessing GPT4-V on Structured Reasoning Tasks](https://arxiv.org/html/2312.11524v1#:~:text=Figure%C2%A03%20compares%20the%20corresponding%20prompts,step%20fashion)) ([Assessing GPT4-V on Structured Reasoning Tasks](https://arxiv.org/html/2312.11524v1#:~:text=,Prompting)).
- **Multimodal eval of GPT-4V**: Some papers evaluate GPT-4V on niche tasks, like understanding scientific diagrams or doing **infographics QA**. Often GPT-4V does very well, but interestingly one study found on a highly abstract visual reasoning test (Chollet’s ARC with images), GPT-4V did worse than GPT-4 text-only + a human-written description of the image ([Assessing GPT4-V on Structured Reasoning Tasks](https://arxiv.org/html/2312.11524v1#:~:text=We%20find%20that%20for%20both,the%20latter%20of)). This indicates that for certain problems, breaking the task into “describe image then use text model” might still be better – an area for future improvement.

### 3.5 A Unified Future: Multimodal Reasoning and Acting  
Having covered reasoning, tools, and vision separately, it’s worth noting that the long-term trend is to **combine these advancements**. For instance:
- An ideal assistant might take in text and images, think step-by-step, use tools when needed, and produce answers.
- GPT-4 already combines vision and reasoning (we saw it can do CoT on images in research).
- One can integrate tool use in multimodal models as well (e.g., an agent that can see an image and decide to call an OCR API or a specific vision tool – though GPT-4V doesn’t need OCR API since it has it built-in).
- Robotics is a domain where all three converge: say a robot with a camera (vision) using an LLM to plan (reasoning) and act (control robot or query tools). There are research projects like PaLM-E (Google, 2023) which integrated a language model with embodied control, allowing an LLM to process visual observations from a robot and output actions in the form of navigation or manipulation commands.

**Apple’s MMXL (hypothetical)**: If MM1 is extended, perhaps they or others will include audio, spatial data, etc., making *multi*-multimodal models. The trend is moving from LLM to **LLM++** (language + other modalities, plus reasoning and tool interfaces).

## Conclusion  
Transformer-based LLMs have evolved from pure text predictors to **general problem solvers**. We examined three cutting-edge dimensions of this evolution:

- **Reasoning LLMs**: By leveraging chain-of-thought prompting and related techniques, LLMs can perform complex reasoning tasks previously out of reach, achieving far better results on benchmarks like GSM8K and ARC ([A Comprehensive Guide to Chain-of-Thought Prompting - Future Skills Academy](https://futureskillsacademy.com/blog/chain-of-thought-prompting/#:~:text=You%20can%20notice%20examples%20of,tuning%20or%20special%20training)). This makes them more reliable and transparent in logical domains.

- **Autonomous/Tool-Using Agents**: Giving LLMs the ability to use tools and act in a loop transforms them into interactive agents. They can fetch information, run computations, and perform multi-step workflows on their own, greatly extending their capabilities beyond what’s stored in their parameters ([[2302.04761] Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761#:~:text=achieve%20the%20best%20of%20both,with%20much%20larger%20models%2C%20without)). Frameworks like ReAct ([ReAct Prompting | Prompt Engineering Guide<!-- --> ](https://www.promptingguide.ai/techniques/react#:~:text=ReAct%20is%20a%20general%20paradigm,involved%20to%20perform%20question%20answering)) demonstrate how reasoning and acting together yield more powerful systems, and projects like AutoGPT hint at the potential (and challenges) of AI agents pursuing open-ended goals.

- **Vision-Language Models**: Integrating visual understanding with language allows AI to interpret and describe the world in rich detail. Modern VLMs can caption images, answer visual questions, and even reason about visual content, powered by advances like CLIP’s representations ([CLIP: Connecting text and images | OpenAI](https://openai.com/index/clip/#:~:text=We%20present%20a%20neural%20network,image)) and huge multimodal models like GPT-4V. This brings AI closer to human-like perception and understanding, enabling applications from aiding the visually impaired to analyzing scientific figures.

These advancements do not exist in isolation – the most exciting systems combine all three. For example, a medical assistant AI might look at a patient’s X-ray (vision), reason through a diagnosis (CoT), and consult medical databases or calculators (tools) before giving an answer. Each component we discussed adds a layer of capability:
- **Reasoning** gives depth (the “thinking” skill),
- **Agents/Tools** give breadth and action (the “doing” skill),
- **Vision multimodality** gives perception (the “seeing” skill).

Together, they are pushing AI toward more **general intelligence** – systems that can perceive, think, and act.

The research landscape in 2024-2025 is incredibly active. Notable papers like *Toolformer*, *ReAct*, *PaLM-E*, *Flamingo*, *BLIP-2*, *GPT-4 Technical Report*, *MM1*, etc., mark the milestones we discussed, and new ones are emerging constantly. Benchmarks continue to get tougher, and models continue to rise to the challenge – often rapidly outpacing prior state-of-the-art within months.

For a graduate student studying these topics, key takeaways are:
- Prompt engineering and clever use of LLMs (like CoT and ReAct) can dramatically improve performance without changing model architecture.
- There is a trend towards **interactivity** – making LLMs active agents rather than passive answerers.
- Multimodality is breaking the barrier between text and the rest of the world, which will unlock far more applications (and also requires multidisciplinary knowledge of vision, NLP, etc.).
- Scale is not the only path; many of these advances achieve more by **using models smarter** (e.g., a 6B Toolformer doing what a 175B couldn’t because it lacked tools).

In conclusion, the progress in these three areas – reasoning, agents, and VLMs – represents a significant step change in what AI systems can do. They are more **intelligent** in a practical sense: they can reason through hard problems, take actions to get information or affect the world, and understand multiple modalities. As research continues, we can expect future LLM-based systems to seamlessly integrate all these abilities, bringing us closer to AI that can see, think, and act in the world much like an human assistant would (albeit with superhuman knowledge and speed in certain aspects). It’s an exciting time, and the lines between “language model” and “general AI agent” are increasingly blurring.