## I have been working on a python library with a react framework that is designed to extract tables from pdfs. Specifically, financial credit card
statements. There are a number of issues I had with existing frameworks that I am attempting to solve here. 

1. Difficult to alter settings or understand how parameters impact efficacy of extractions.
2. None of them truly just work out of the box, or have a rate of success over 70% in my tests.
3. None of them validate output -- does the sum of this column equal the totals row?
-- The way i have designed this places an emphasis on raw logic, extraction, and rules based approach to 
extracting tables from documents. We can be fairly certain when we have extracted something correctly, but
we can never be certain that we didnt miss something or mess something up logically. Thats where the LLM inference 
comes into play. When in doubt, ask a vision model to verify. I am finding that this combination yields very high 
quality extractions.
4. Over the last year I have made MANY iterations of this library, often just copying a folder and starting over. In each on there 
are solid functions, logic, utilites, etc. 

You will have 4 tools that you are able to use in solving your main task. -- Iterate through all older versions of this library, consolidating modules, logic 
and tools -- and updating my final, co-te-gra -- version of this library. 

You will run in a loop until the task is complete: get the react website running again, with all endpoints restored for running pdf table extraction processes
alongside extraction validations, bounding box visualizations and displays in the react site. 

The tools at your disposal are:

1. Read directory (containing all older versions of my projects.)
2. Read file -- read in full content of a given file.
3. Write output (update code in final repo.)
4. Deploy docker compose of the application (test)
5. Retrieve screenshots of the website.


# Composable Design Patterns in Python Query Frameworks
This architectural analysis explores the **generic and composable design patterns** in Meta/Facebook's async query execution framework and similar systems, revealing how simple primitives compose into sophisticated data processing engines.
## The architecture of composition

Modern query frameworks achieve power not through complexity but through elegant composition of simple primitives. Meta's async query execution framework exemplifies this philosophy, where **async generators**, **visitor patterns**, and **generic base classes** combine to create flexible, performant systems that can target multiple execution backends while maintaining type safety and composability.

The framework's architecture centers on three key principles: **lazy evaluation** for memory efficiency, **generic typing** for reusable components, and **monadic composition** for clean async operations. These patterns enable building query engines that process both structured and unstructured data at massive scale, from Facebook's GraphQL infrastructure to LinkedIn's knowledge graph.

## Async iteration as the foundation
### Building blocks of async composition

The async iteration utilities form the bedrock of composable query systems. Rather than loading entire datasets into memory, async generators enable **stream processing** with constant memory usage:

```python
# Core async iterator composition pattern
class AsyncIteratorBase:
    """Base pattern for composable async iterators"""
    
    async def __aiter__(self):
        return self
    
    async def __anext__(self):
        raise StopAsyncIteration
    
    def map(self, func):
        """Chainable transformation"""
        return AsyncMapIterator(self, func)
    
    def filter(self, predicate):
        """Chainable filtering"""
        return AsyncFilterIterator(self, predicate)

# Composable async operations
async def async_pipeline():
    async def numbers():
        for i in range(1000000):  # Large dataset
            yield i
            await asyncio.sleep(0)  # Yield control
    
    # Memory-efficient composition
    result = (AsyncIteratorBase(numbers())
              .filter(lambda x: x % 2 == 0)
              .map(lambda x: x * x)
              .take(100))  # Early termination
    
    async for item in result:
        process(item)  # Process one item at a time
```

This pattern demonstrates **lazy evaluation** - operations don't execute until consumption begins. Each transformation returns a new iterator that wraps the previous one, creating a **pipeline of operations** evaluated on-demand.

### Monadic patterns in async programming

Python's async/await implements **monadic composition** where async functions return awaitable objects (monadic values) and await performs monadic bind:

```python
class AsyncResult(Generic[T]):
    """Async monad for composable operations"""
    
    def __init__(self, coro):
        self.coro = coro
    
    async def bind(self, func: Callable[[T], 'AsyncResult[U]']) -> 'AsyncResult[U]':
        """Monadic bind for async operations (>>= in Haskell)"""
        result = await self.coro
        next_async = func(result)
        return await next_async.coro
    
    async def map(self, func: Callable[[T], U]) -> 'AsyncResult[U]':
        """Functor map for async operations"""
        result = await self.coro
        return AsyncResult(async_coroutine(func(result)))

# Kleisli composition for async operations
class KleisliAsync:
    """Kleisli category for async composition"""
    
    def __init__(self, func: Callable[[A], Awaitable[B]]):
        self.func = func
    
    def compose(self, other: 'KleisliAsync') -> 'KleisliAsync':
        """Compose: (A -> M[B]) ∘ (B -> M[C]) = (A -> M[C])"""
        async def composed(a):
            b = await self.func(a)
            return await other.func(b)
        return KleisliAsync(composed)

# Usage: Meta-style async query composition
fetch_user = KleisliAsync(lambda id: database.get_user(id))
load_permissions = KleisliAsync(lambda user: permissions.get(user.id))
validate = KleisliAsync(lambda perms: validator.check(perms))

auth_pipeline = fetch_user.compose(load_permissions).compose(validate)
```

This monadic composition enables **error handling without nested try-catch blocks** and **nullable result handling without conditionals**, following category theory principles where queries form a category with composition.

## The visitor pattern for generic traversal

### AST visitor architecture

The visitor pattern enables **generic traversal** of query structures, allowing the same query to compile to different backends:

```python
class QueryASTVisitor(ast.NodeVisitor):
    """Generic visitor for query AST traversal"""
    
    def __init__(self, backend: str):
        self.backend = backend
        self.context = {}
        self.transformations = []
    
    def visit(self, node: ast.AST) -> Any:
        """Double dispatch based on node type"""
        method_name = f'visit_{node.__class__.__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def visit_Call(self, node):
        """Transform function calls for target backend"""
        if self.backend == 'sql':
            return self._transform_to_sql(node)
        elif self.backend == 'malloy':
            return self._transform_to_malloy(node)
        elif self.backend == 'polars':
            return self._transform_to_polars(node)

# Natural transformation between backends
class BackendTransformer:
    """Natural transformation preserving query structure"""
    
    def transform(self, ast_node, from_backend, to_backend):
        # Extract semantic meaning
        semantic = self.extract_semantics(ast_node, from_backend)
        # Reconstruct for target backend
        return self.reconstruct(semantic, to_backend)
```

This visitor pattern relates to **F-algebras** in category theory, where the visitor defines an algebra over the syntax tree functor. Each visit method provides a **structure-preserving transformation** that maintains query semantics across backends.

### Compilation target abstraction

The framework abstracts compilation targets through a **strategy pattern** combined with visitors:

```python
class QueryBackend(ABC):
    """Abstract backend for query compilation"""
    
    @abstractmethod
    async def compile_query(self, ast_node) -> str:
        pass
    
    @abstractmethod
    async def execute_query(self, compiled_query: str) -> Any:
        pass

class VeloxBackend(QueryBackend):
    """Meta's Velox vectorized execution"""
    
    async def compile_query(self, ast_node) -> str:
        # Transform to columnar + vectorized operations
        visitor = VeloxVisitor()
        return visitor.visit(ast_node)
            .withColumnarLayout()
            .withVectorizedExecution()
            .withLazyMaterialization()
            .optimize()
```

## Composable query building

### The FuncStack pattern

FuncStack enables **function composition** for building complex operations from simple ones:

```python
class FuncStack:
    """Composable function stack for query operations"""
    
    def __init__(self, operations=None):
        self.operations = operations or []
    
    def then(self, func):
        """Chain operations functionally"""
        return FuncStack(self.operations + [func])
    
    def compose(self, other):
        """Compose with another FuncStack"""
        return FuncStack(self.operations + other.operations)
    
    async def execute(self, data):
        """Execute composed operations with async support"""
        result = data
        for op in self.operations:
            if asyncio.iscoroutinefunction(op):
                result = await op(result)
            else:
                result = op(result)
        return result

# Transducer pattern for efficient composition
class Transducer:
    """Single-pass data transformation"""
    
    def __init__(self, transformer):
        self.transformer = transformer
    
    def compose(self, other):
        """Compose transducers for single-pass processing"""
        return Transducer(lambda reducer: 
                         self.transformer(other.transformer(reducer)))

# Usage: Compose complex query pipeline
query_stack = (FuncStack()
    .then(filtering(lambda x: x.active))
    .then(mapping(lambda x: x.to_dict()))
    .then(batching(size=100))
    .then(async_enrichment))

result = await query_stack.execute(data_stream)
```

This pattern demonstrates how **simple primitives** (map, filter, batch) compose into complex behaviors through **higher-order functions**.

### The @nested decorator pattern

The @nested decorator creates **composable transformations** that wrap and modify query behavior:

```python
def nested(transformation_func: Callable):
    """Decorator for nested query transformations"""
    
    def decorator(query_func):
        @wraps(query_func)
        async def wrapper(*args, **kwargs):
            # Execute original query
            result = await query_func(*args, **kwargs)
            
            # Apply nested transformation
            if hasattr(result, '__aiter__'):
                # Handle async iterables
                async def transform_async():
                    async for item in result:
                        yield await transformation_func(item)
                return transform_async()
            else:
                # Handle regular results
                return await transformation_func(result)
        
        return wrapper
    return decorator

# Compose transformations through decoration
@nested(lambda x: x.group_by('category').aggregate('sum'))
@nested(lambda x: x.filter(amount > 100))
async def sales_query():
    return await execute_base_query()
```

### Comparison with other frameworks

The patterns mirror approaches in established frameworks:

**Pydantic**: Uses generic types (`BaseModel[T]`) and validator composition similar to the framework's type parameters and transformation chains.

**SQLAlchemy**: Method chaining and lazy evaluation parallel the framework's FuncStack and deferred execution patterns.

**FastAPI**: Dependency injection resembles the framework's composable resolver chains.

**Polars**: Expression API with lazy evaluation matches the framework's expression trees and lazy query execution.

## Practical applications

### Query engines for unstructured data

The composable patterns enable sophisticated query engines like **LOTUS** (LLMs Over Tables of Unstructured and Structured data):

```python
# Semantic query operations compose with traditional operations
result = (dataframe
    .sem_index("content", index_path)  # Vector indexing
    .sem_search("content", "machine learning", k=100)  # Semantic search
    .filter(lambda x: x.date > "2023-01-01")  # Traditional filter
    .sem_agg("content", "summarize key points"))  # LLM aggregation
```

### Knowledge graph querying

The patterns enable efficient knowledge graph queries across different backends:

```python
# Unified interface for different graph databases
class UniversalGraphQuery:
    def __init__(self, backend):
        self.backend = backend  # Neo4j, ArangoDB, SPARQL endpoint
        self.operations = []
    
    def match(self, pattern):
        self.operations.append(('match', pattern))
        return self
    
    def where(self, condition):
        self.operations.append(('where', condition))
        return self
    
    async def execute(self):
        # Compile to backend-specific query
        if self.backend == 'neo4j':
            cypher = self.to_cypher()
            return await self.backend.run(cypher)
        elif self.backend == 'sparql':
            sparql = self.to_sparql()
            return await self.backend.query(sparql)
```

## Building similar composable systems

### Design principles for composability

When building composable systems, follow these architectural principles:

1. **Small, focused interfaces**: Each component should do one thing well
2. **Immutable operations**: Return new instances rather than modifying state
3. **Type parameters**: Use generics for reusable components
4. **Lazy evaluation**: Defer execution until necessary
5. **Composition over inheritance**: Build complex behavior through composition

### Implementation blueprint

```python
# Template for composable query system
class ComposableQuerySystem:
    """Blueprint for building composable query engines"""
    
    def __init__(self):
        self.operations = []
        self.optimizations = []
        self.cache = {}
    
    def add_operation(self, op):
        """Immutable operation addition"""
        new_system = copy(self)
        new_system.operations.append(op)
        return new_system
    
    def optimize(self):
        """Apply optimization rules"""
        # Predicate pushdown
        # Operation fusion
        # Parallel execution planning
        return optimized_system
    
    async def execute(self, backend):
        """Execute with specific backend"""
        plan = self.optimize()
        compiled = backend.compile(plan)
        return await backend.execute(compiled)
```

## Performance and scalability patterns

### Memory efficiency through streaming

The framework achieves **constant memory usage** regardless of dataset size through streaming:

```python
# Process terabytes with megabytes of memory
async def process_large_dataset():
    async for batch in stream_from_database(batch_size=1000):
        processed = await transform_batch(batch)
        await write_results(processed)
        # Memory usage remains constant
```

### Query optimization through composition

Composable operations enable **automatic optimization**:

```python
class QueryOptimizer:
    def push_down_predicates(self, operations):
        """Move filters closer to data source"""
        filters = [op for op in operations if op.type == 'filter']
        others = [op for op in operations if op.type != 'filter']
        return filters + others  # Filters execute first
    
    def fuse_operations(self, operations):
        """Combine compatible operations"""
        # Map(f) -> Map(g) becomes Map(compose(g, f))
        fused = []
        for op in operations:
            if fused and can_fuse(fused[-1], op):
                fused[-1] = fuse(fused[-1], op)
            else:
                fused.append(op)
        return fused
```