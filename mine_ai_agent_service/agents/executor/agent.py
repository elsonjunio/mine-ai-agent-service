from dataclasses import dataclass, field

from langgraph.graph.state import CompiledStateGraph

from mine_ai_agent_service.agents.graph_builder.builder import PipelineState
from mine_ai_agent_service.agents.planner.agent import Plan, PlanStep


@dataclass
class StepResult:
    index: int
    agent: str
    task: str
    output: str


@dataclass
class ExecutionResult:
    request: str
    steps: list[StepResult] = field(default_factory=list)

    @property
    def final_output(self) -> str:
        return self.steps[-1].output if self.steps else ''

    @property
    def all_outputs(self) -> str:
        return '\n\n'.join(
            f'[Passo {s.index + 1} — {s.agent}]\nTarefa: {s.task}\n{s.output}'
            for s in self.steps
        )


class ExecutorAgent:
    """Executa o graph compilado e monitora cada etapa via streaming."""

    def run(
        self,
        graph: CompiledStateGraph,
        request: str,
        plan: Plan,
        context: dict[str, str] | None = None,
    ) -> ExecutionResult:
        result = ExecutionResult(request=request)
        accumulated: list[str] = []

        initial_state: PipelineState = {
            'request': request,
            'context': context or {},
            'results': [],
        }

        for chunk in graph.stream(initial_state, stream_mode='updates'):
            # chunk = {node_name: {field: value, ...}}
            node_name, update = next(iter(chunk.items()))
            new_results: list[str] = update.get('results', [])
            if not new_results:
                continue

            step_index = len(accumulated)
            plan_step: PlanStep = plan.steps[step_index]
            output = new_results[-1]
            accumulated.append(output)

            step_result = StepResult(
                index=step_index,
                agent=plan_step.agent,
                task=plan_step.task,
                output=output,
            )
            result.steps.append(step_result)
            self._log_step(step_result)

        return result

    @staticmethod
    def _log_step(step: StepResult) -> None:
        separator = '─' * 60
        print(f'\n{separator}')
        print(f'[Step {step.index + 1}] Agente: {step.agent}')
        print(f'Tarefa : {step.task}')
        print(f'Output :\n{step.output}')
        print(separator)
