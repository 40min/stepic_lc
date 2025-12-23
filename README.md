# Stepic LC - LangChain Learning Project

A Python project demonstrating LangChain usage with console chat bots and prompt engineering examples.

## Features

- **Console Chat Bot**: Interactive CLI bot using OpenRouter API with conversation history
- **Prompt Engineering Examples**: Various LangChain chain patterns and implementations
- **JSON Output Validation**: Using Pydantic models for structured responses

## Project Structure

```
stepic_lc/
├── src/
│   ├── 1-console-chat-bot/
│   │   ├── bot.py          # Main CLI bot implementation
│   │   └── chains.py       # Basic LangChain chain examples
│   ├── 2-prompt-engineering/
│   │   ├── chain_on_messages.py  # Complex chain with message handling
│   │   └── valid_json_out.py     # JSON output validation with Pydantic
│   └── 3-rag/
│       ├── chunk_sizes/
│       │   ├── chunker.py        # RAG chunking strategy evaluator
│       │   ├── llm_assessor.py  # LLM-based chunk quality assessment
│       │   └── evaluators.py    # Evaluation strategies
│       └── eval_test/
│           ├── eval.py          # Comprehensive RAG evaluation system
│           └── eval_data.py     # Test data and ground truth for evaluation
├── pyproject.toml          # Project configuration and dependencies
├── Makefile               # Build and run commands
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone or navigate to the project directory:
   ```bash
   cd stepic_lc
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Set up environment variables by creating a `.env` file:
   ```bash
   cp .env.example .env  # If you have an example file
   # Or create .env manually with required variables
   ```

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_API_MODEL=your_preferred_model_name

# Alternative OpenAI Configuration (if using OpenAI directly)
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_API_MODEL=gpt-3.5-turbo  # or your preferred model
```

## Usage

### 1. Console Chat Bot

Run the interactive CLI bot:
```bash
python src/1-console-chat-bot/bot.py
```

**Features:**
- Interactive conversation with memory
- Russian language support
- Session management with unique session IDs
- Command shortcuts:
  - `выход`/`стоп`/`конец` - Exit the bot
  - `сброс` - Clear conversation context
- Logging to `chat_session.log`

### 2. Prompt Engineering Examples

Run individual examples to explore LangChain patterns:

```bash
# Basic chain examples
python src/1-console-chat-bot/chains.py

# Complex chain with message handling  
python src/2-prompt-engineering/chain_on_messages.py

# JSON output validation
python src/2-prompt-engineering/valid_json_out.py
```

### 3. RAG Chunking Strategy Evaluation

Evaluate different chunking strategies for RAG systems using score-based or LLM-based assessment:

```bash
# Score-based evaluation (default, fast, uses FAISS similarity)
make chunker
# or
uv run python src/3-rag/chunk_sizes/chunker.py

# LLM-based evaluation (slower, provides reasoning, requires API key)
make chunker-llm
# or
uv run python src/3-rag/chunk_sizes/chunker.py --eval-mode llm-based

# Use specific LLM model
uv run python src/3-rag/chunk_sizes/chunker.py --eval-mode llm-based --llm-model anthropic/claude-3-opus
```

**Evaluation Modes:**
- **score-based**: Fast FAISS distance-based evaluation (semantic similarity)
- **llm-based**: LLM assesses chunk usefulness with reasoning (requires `OPENROUTER_API_KEY`)

**Features:**
- Tests multiple chunking configurations (sparse, dense, hybrid)
- Evaluates retrieval quality on 17 test questions about Indonesia
- Provides winner statistics and sample outputs
- LLM mode includes reasoning for each assessment

### 4. RAG System Evaluation

Comprehensive evaluation system for RAG systems with both retrieval metrics and answer quality assessment:

```bash
# Run complete evaluation (retrieval + LLM-based answer quality)
make eval
# or
uv run python src/3-rag/eval_test/eval.py
```

**Evaluation Components:**
- **Retrieval Metrics**: Precision@K and Recall@K calculation for document retrieval
- **LLM-based Assessment**: Answer quality evaluation using ChatOpenAI
- **Ground Truth Comparison**: Compares predicted answers against golden standard
- **Russian Test Data**: Tea-related questions and documents for evaluation

**Features:**
- Tests retrieval performance on 12 tea-related questions
- Uses 7 Chinese tea documents with ground truth mappings
- LLM assessment evaluates answer correctness with detailed scoring
- Bilingual evaluation (metrics in Russian, technical assessment in English)
- Requires `OPENROUTER_API_KEY` for LLM-based answer quality assessment

**Default Configuration:**
- K=2 for precision/recall calculations
- OpenRouter API with configurable model (default: x-ai/grok-4-fast)
- HuggingFace embeddings (cointegrated/rubert-tiny2)

## Dependencies

### Core Dependencies
- **langchain**: Main LangChain framework
- **langchain-core**: Core LangChain components
- **langchain-openai**: OpenAI integration for LangChain
- **openai**: OpenAI Python client
- **python-dotenv**: Environment variable management
- **pydantic**: Data validation and settings management

### Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatting
- **ruff**: Fast Python linter

## Key Concepts Demonstrated

### 1. **Basic Chains**
- Simple prompt → LLM chains using LCEL (LangChain Expression Language)
- RunnablePassthrough, RunnableLambda usage
- String output parsing

### 2. **Advanced Chain Patterns**
- RunnableParallel for concurrent execution
- RunnableBranch for conditional logic
- Complex chain composition

### 3. **Memory Management**
- InMemoryChatMessageHistory for conversation storage
- Session-based context management
- RunnableWithMessageHistory for stateful conversations

### 4. **Structured Output**
- Pydantic integration for JSON validation
- Custom output parsers
- Format instruction generation

### 5. **Error Handling**
- API timeout and connection error handling
- Authentication error management
- User-friendly error messages in Russian

## Development

### Code Quality

Format code with:
```bash
black src/
```

Lint with:
```bash
ruff src/
```

### Testing

Run tests:
```bash
pytest
```

## API Configuration

### OpenRouter Setup
1. Get your API key from [OpenRouter](https://openrouter.ai/)
2. Choose a model (e.g., "anthropic/claude-3-haiku", "openai/gpt-3.5-turbo")
3. Set the environment variables as shown above

### OpenAI Setup (Alternative)
1. Get your API key from [OpenAI](https://platform.openai.com/)
2. Choose a model (e.g., "gpt-3.5-turbo", "gpt-4")
3. Update the code to use OpenAI endpoints instead of OpenRouter

## Logging

The chat bot automatically logs interactions to `chat_session.log` with timestamps and user/bot messages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run linting and formatting
6. Submit a pull request

## License

MIT License - feel free to use this project for learning and development.

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenRouter API](https://openrouter.ai/docs)
- [OpenAI API](https://platform.openai.com/docs)
- [uv Documentation](https://docs.astral.sh/uv/)