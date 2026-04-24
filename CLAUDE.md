# CLAUDE.md

Refer to [AGENTS.md](methods/evermemos/AGENTS.md) for comprehensive project documentation including:
- Project architecture and structure
- Tech stack and dependencies
- Code conventions and patterns
- Key abstractions and files
- Development guidelines
- Database schema

## Quick Commands

```bash
cd methods/evermemos
docker-compose up -d          # Start infrastructure
uv sync                       # Install dependencies
make run                      # Run application
pytest                        # Run tests
black src/ && isort src/      # Format code
pyright                       # Type check
```

## Key Entry Points

- `methods/evermemos/src/run.py` - Application entry
- `methods/evermemos/src/agentic_layer/memory_manager.py` - Core memory manager
- `methods/evermemos/src/infra_layer/adapters/input/api/` - REST API controllers
- `methods/evermemos/src/infra_layer/adapters/out/search/leann/` - Optional LEANN local search backend

## Remember

- All I/O is async - use `await`
- Multi-tenant system - data is tenant-scoped
- Prompts in `methods/evermemos/src/memory_layer/prompts/` (EN/ZH)
- Set `MEMORY_SEARCH_BACKEND=leann` to route episodic / atomic fact / foresight search indexes to LEANN
