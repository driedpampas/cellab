# Deliverable 1: Speaker script

Spoken script for `CelLab-Deliverable1.pdf` (35 slides).

- **Length:** about 12 minutes at a calm pace. If you are told 5–7 minutes, say only the slides marked **[CORE]** and click past the others with one sentence.
- **Split:** Speaker A does slides 1–14 (problem, automata, design). Speaker B does 15–35 (questions, AI, plan). Swap the names if you prefer.
- **Rule of thumb:** don't read the slide out loud. Say the one idea, point at the picture, move on. The tables are there for the professor to read, not for you to recite.

---

## Speaker A: the problem and the design

**1. Cover [CORE]**
> Hello, we are Bianca Alexandru and Matei Gurzu, and our project is CelLab. It is a system where you describe a cellular automaton in plain English, it simulates it, and it answers questions about how it evolves. Today we present our team and our technical plan.

**2. What the system should do [CORE]**
> The system does four things. The user **describes** an automaton in English. We **simulate** it for N generations. We **answer** the five questions from the brief exactly, using the simulation. And as an extra, a trained model tries to **predict** those answers without simulating, and we measure how good it is. All of this sits behind one graphical interface.

**3. Architecture at a glance [CORE]**
> Here is how the parts connect. On the main path, the English text goes through a parser and validator and becomes a spec file in JSON. The engine runs that spec and gives exact answers. On the prediction path, the engine generates many runs, which become a dataset for a neural network. The orange boxes are the AI parts: the language parsing, and the network. Everything else is ordinary code.

**4. Cellular automata in one slide**
> Quick reminder of what a cellular automaton is. A grid of cells, a set of cell types, a neighbourhood, and a rule that every cell applies at the same time. In forest fire, for example: fire burns out, and any tree touching fire catches fire. You can see one step here.

**5. Elementary automata**
> The simplest family is the elementary automata: one row of 0s and 1s, and each cell looks at itself and its two neighbours. That is 8 patterns, each mapped to 0 or 1, so there are exactly 256 rules. These 8 bits are useful for us later, because they are a ready-made way to give a rule to a neural network.

**6. The 15 examples [CORE]**
> The brief gives 15 example automata. We see them as a test set, not a list to hard-code: the system has to be general. We grouped them into four kinds: rules based on neighbours, cells that hold numbers, things that move, and agents with their own memory.

**7. Difficulty map**
> We estimated how hard each one is to simulate, to answer questions about, and to learn. The hardest are the agent-based ones, like predator–prey and the ecosystem, and anything random.

**8. Scope [CORE]**
> This is what we will support: 1D and 2D grids, any number of cell types or numeric values, 4 or 8 neighbours, edges that are fixed or wrap around, movement, simple agents, and randomness with a seed. Just as important is the right column: what the validator catches. An undeclared cell type, contradicting rules, missing information. If something cannot be expressed, we refuse it with a reason. We never guess.

**9. The spec file [CORE]**
> The central idea is one spec format. This is Brian's Brain as a JSON file: the states, the neighbourhood, the edges, and the rules. The engine only reads files like this, so the same code runs every automaton. And in Deliverable 4, the language parser will produce exactly this format.

**10. Representation options**
> We considered other options: a Python class per automaton, a custom rule language, or letting a language model write code. We chose JSON with fixed rule types because it is general and it can be checked. The parser needs a clear target, and the engine must run new automata without new code.

**11. Update modes**
> Not every automaton updates the same way. Most are synchronous: every cell updates at once. Traffic and falling sand need ordered movement with a collision rule. Langton's ant and predator–prey need agents. All three modes share the same storage, edges, seeds and cycle detection.

**12. Engine options**
> For the engine: NumPy for the synchronous mode, because counting neighbours is a convolution and it is fast. Numba for movement and agents, where cells must be processed one by one.

**13. Initial world**
> The starting grid can come from four places: drawn by hand, random with chosen proportions, a known pattern, or from the description itself, like "trees everywhere, one fire in the middle". The random seed is always saved, so any start can be recreated.

**14. State space and storage**
> The number of possible states is huge: 2 to the 64 for a 64-cell row, and around 10 to the 1192 for a 50 by 50 grid with 3 types. So we never list states. We store grids compactly, we save experiments as spec plus parameters plus seed so they can be replayed, and storing every generation is optional, as the brief asks.

> **Handover:** *"Now Matei will explain how we answer the five questions."* (or the other way round)

---

## Speaker B: questions, AI, and the plan

**15. The five questions [CORE]**
> These are the five questions from the brief. Our goal is all five, for every supported automaton. The key fact is on the right: a finite grid with fixed rules has finitely many states. So every run must eventually repeat a state, and from then on it loops forever. That makes questions 2, 3 and 5 exactly answerable.

**16. Cycle detection [CORE]**
> Here is how. We keep a table from a hash of each state to the step where we saw it. The first time a state repeats, we know the run-up and the cycle length. That answers question 5. A stable state is just a cycle of length 1, which answers question 2. And if a configuration has not appeared by the time the loop closes, it never will, which answers question 3.

**17. Three ways to answer**
> We answer in three ways. Simulation always works and is the ground truth. Rule properties give instant shortcuts in some cases: traffic never changes its number of cars, crystals only grow. And the trained model predicts answers instantly but approximately. The interface will show which method gave each answer.

**18. Question-answering options**
> We compared approaches. Simulating to a step limit cannot prove "never". Storing hashes is exact and simple, so that is our choice. If a cycle is too long to keep in memory, we switch to Brent's method, which uses almost no memory.

**19. Parsing natural language [CORE]**
> The first AI job is Deliverable 4: turning English into a spec. A language model writes the spec, and the validator checks it. If there are errors, they go back to the model to fix. If information is missing, the system asks the user. For example, here the user did not say which neighbours count, so the system asks: 4 sides, or all 8?

**20. Parsing options**
> We will compare three options on the same test descriptions: hand-written grammar rules, a large model through an API, and a small model running locally with Ollama. We measure how many specs are valid, whether they behave correctly on 100 random grids, and the time per description.

**21. Building the dataset [CORE]**
> The second AI job is Deliverable 3. Our own simulator generates the dataset, so the labels are exact and as many as we want. Each row is one run: a rule, a start state, a number of steps, and the true answers. These example rows come from our prototype code.

**22. The model**
> The network takes the rule as 8 bits, the start state, N, and a target configuration. A shared convolutional encoder, convolutional because the rules are local, feeds five outputs, one per question. We compare it with five separate networks and with simple baselines.

**23. Encoding**
> This slide is how we turn grids into network inputs: one layer per cell type, numbers scaled to 0–1, and padding that matches the edges. After the 1D rules, we repeat the experiment on 2D automata such as forest fire.

**24. ML options**
> We considered gradient boosting, fully connected, recurrent and transformer networks, and reinforcement learning. We chose a convolutional network, with gradient boosting as a baseline. The open question from our notes, one shared network or five, we will answer by training both.

**25. Metrics**
> Every question gets its own metric: share of cells correct for the grid, F1 for yes/no, average error for counts. Everything is reported per class of rule, on data the model has never seen, next to the baselines.

**26. Hypothesis [CORE]**
> Our hypothesis: the model will be near perfect on simple rules that die out, good on periodic rules, poor on complex ones like Rule 110, and close to chance on chaotic ones like Rule 30. We want to measure where prediction breaks.

**27. Rule 30 [CORE]**
> Rule 30 is the hard case. There is a prize with three open questions about it, and the third one, can you compute step n faster than simulating n steps, is exactly our "answer without running" idea. We tried it on small rings with one live cell. It must cycle, but look at the cycle lengths: 40, 102, 1,428, 6,016, 10,846, then 247. No visible pattern.

**28. Interface options**
> For the interface, the brief requires a text box, the number of generations, the initial state, the option to store generations, and a viewer. We are leaning towards Streamlit because it is the fastest to build. We will decide in Deliverable 2.

**29. Full example [CORE]**
> Here is the whole idea on one example. The user describes forest fire in English. That becomes a spec. And the system answers: after 2 steps there are 8 trees and 5 fires, it is stable from step 5 when everything has burned, it ends in a cycle of length 1, and a grid with trees but no fire never occurs.

**30. Components and timeline**
> Each component has a deliverable. For Deliverable 2: the spec, validator, engine, storage and question engine. Deliverable 3: dataset and model. Deliverable 4: the language parser. The interface grows at every step.

**31. Tools and hardware**
> Everything is in Python: NumPy and Numba for simulation, PyTorch for the network, Ollama for a local model, Streamlit for the interface. Our laptops have no NVIDIA GPU, so we train on Google Colab and run everything else locally.

**32. Risks**
> The main risks: descriptions that leave things out, where we ask or use a documented default. Chaotic rules, where we measure and report the failure as a result. Randomness, where we use seeds and statistics. And scope for two people: general engine first, then the 15 examples, then the bonuses.

**33. Submission**
> At the end we submit the four things the brief requires: the source code with tests, all our AI conversation logs, which are already saved automatically after every reply, the training data with the generator code, and an install and usage guide.

**34. Timeline [CORE]**
> The plan for the semester: today the plan, Lab 6 the engine and the five questions, Lab 11 the dataset and network, Lab 13 the parser, and Lab 14 the final version. The bonuses we aim for are all five questions, the neural network experiment, and Rule 30.

**35. Thank you [CORE]**
> Thank you. We have one question for you as well: will the final evaluation use the 15 examples, or new automata we have not seen? That changes how general the parser needs to be.

---

## Likely questions and short answers

**"Why not just let ChatGPT write the simulation code?"**
> Because we couldn't check it, and running generated code is unsafe. A JSON spec can be validated, and one engine runs all of them.

**"How do you know two states are the same?"**
> We hash the whole grid. To be safe from collisions, when two hashes match we compare the actual grids before calling it a cycle.

**"What if the cycle never comes in practice?"**
> On a big grid it can take too long. Then we stop at a step limit and say "no cycle found up to step N", not "no cycle".

**"What about random automata, they never really cycle?"**
> With a fixed seed the run is repeatable, so we answer for that seed. Otherwise we run many seeds and give statistics, for example "stable in 80% of runs".

**"Why a neural network if you can simulate exactly?"**
> The brief rewards a dataset and NN experiment. The interesting result is *where* prediction works and where it fails, and Rule 30 shows it can fail.

**"Who does what?"**
> Roles aren't fixed yet. We will split by component when Deliverable 2 starts, for example one of us on engine and questions, the other on storage and interface.

**"Did you write code already?"**
> Only a small prototype for the numbers on the slides (dataset rows, Rule 30 table, forest fire example). The real implementation starts with Deliverable 2.

**"How do you handle Langton's ant, it's not a normal automaton?"**
> It's our agent mode: the ant is an agent with a position and a direction, and the grid is its world. Same storage and cycle detection; the ant's position is part of the state.

**If you don't know:** "Good question, we haven't decided that yet. We'll cover it in Deliverable 2." That is an acceptable answer at the planning stage.
