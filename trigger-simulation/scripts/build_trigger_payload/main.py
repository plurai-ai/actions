import argparse
import json
import os
import re
import sys
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, BeforeValidator, ConfigDict, Field
from pydantic.alias_generators import to_camel


def empty_to_none(value: Any) -> Any | None:
    if isinstance(value, str) and value.strip() == "":
        return None

    return value


class Args(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    host: str
    scope: str = Field(min_length=1)
    name: str = Field(min_length=1)
    dataset_id: UUID
    eval_set_id: Annotated[UUID | None, BeforeValidator(empty_to_none)] = None
    environment_id: str
    docker_image: str = Field(min_length=1)
    docker_entrypoint: str = Field(min_length=1)
    docker_entrypoint_args: Annotated[str | None, BeforeValidator(empty_to_none)] = None
    docker_env_vars: Annotated[str | None, BeforeValidator(empty_to_none)] = None

    agent_port: int
    agent_name: str

    github_token: Annotated[str | None, BeforeValidator(empty_to_none)] = None
    commit_sha: Annotated[str | None, BeforeValidator(empty_to_none)] = None
    pull_request: Annotated[int | None, BeforeValidator(empty_to_none)] = None


class ApiModel(BaseModel):
    model_config = ConfigDict(
        validate_by_name=True,
        alias_generator=to_camel,
        serialize_by_alias=True
    )


class InlineBlob(ApiModel):
    mime_type: str
    data: Base64Bytes
    metadata: Annotated[dict[str, str], Field(default_factory=dict)]


class GenerationJobDetails(ApiModel):
    name: str
    expected_items: int = 0
    metadata: Annotated[dict[str, str], Field(default_factory=dict)]


class GenerationJobPayload(ApiModel):
    data: dict[str, InlineBlob]
    parameters: dict[str, str]


class CreateGenerationJobRequest(ApiModel):
    details: GenerationJobDetails
    payload: GenerationJobPayload


class DynamicParameter(ApiModel):
    type: Literal['inline']
    value: str


class RemoteSimulationTargetParameters(ApiModel):
    protocol: Literal['langgraph']
    address: str
    agent: str


class RemoteSimulationTarget(ApiModel):
    mode: Literal['remote']
    target_parameters: RemoteSimulationTargetParameters


class DockerSimulationTargetParameters(ApiModel):
    communication_channel: RemoteSimulationTarget
    image_registry: str | None = None
    image_name: str
    image_tag: str = 'latest'
    environment: Annotated[dict[str, DynamicParameter], Field(default_factory=dict)]
    entrypoint: str | None = None
    args: Annotated[list[DynamicParameter], Field(default_factory=list)]


class DockerSimulationTarget(ApiModel):
    mode: Literal['docker']
    target_parameters: DockerSimulationTargetParameters


class ExperimentPayloadParameters(ApiModel):
    dataset_id: str
    eval_set_id: str | None = None
    environment: str
    target: DockerSimulationTarget


class ExperimentPayload(ApiModel):
    scope: str
    name: str
    parameters: ExperimentPayloadParameters
    extra_parameters: Annotated[dict[str, str | dict[str, str]], Field(default_factory=dict)]


def build_payload(args: Args) -> ExperimentPayload:
    registry, image_name, image_tag = parse_docker_image(args.docker_image)

    env_dict = parse_env_vars(args.docker_env_vars) if args.docker_env_vars else {}
    args_list = parse_entrypoint_args(args.docker_entrypoint_args) if args.docker_entrypoint_args else []

    github_params = build_github_params(args)

    return ExperimentPayload(
        scope=args.scope,
        name=args.name,
        parameters=ExperimentPayloadParameters(
            dataset_id=str(args.dataset_id),
            eval_set_id=str(args.eval_set_id) if args.eval_set_id else None,
            environment=str(args.environment_id),
            target=DockerSimulationTarget(
                mode="docker",
                target_parameters=DockerSimulationTargetParameters(
                    communication_channel=RemoteSimulationTarget(
                        mode="remote",
                        target_parameters=RemoteSimulationTargetParameters(
                            protocol="langgraph",
                            address=f"http://localhost:{args.agent_port}",
                            agent=args.agent_name
                        )
                    ),
                    image_registry=registry,
                    image_name=image_name,
                    image_tag=image_tag if image_tag else "latest",
                    environment=env_dict,
                    entrypoint=args.docker_entrypoint,
                    args=args_list
                )
            )
        ),
        extra_parameters={"github": github_params} if github_params else {}
    )


def build_github_params(args: Args) -> dict[str, str] | None:
    if args.github_token is None:
        return None

    repository_str = os.getenv("GITHUB_REPOSITORY")
    if not repository_str:
        raise ValueError("GITHUB_REPOSITORY environment variable is not set")

    owner, repo = parse_github_repository(repository_str)

    pr_number_str = args.pull_request or os.getenv("GITHUB_PR_NUMBER")
    pr_number = int(pr_number_str) if pr_number_str is not None else None

    result: dict[str, str] = {
        "token": args.github_token,
        "owner": owner,
        "repo": repo,
    }

    commit_value = args.commit_sha or os.getenv("GITHUB_SHA")
    if commit_value is not None:
        result["commit"] = commit_value

    if pr_number is not None:
        result["prNumber"] = str(pr_number)

    return result


def parse_github_repository(repository: str) -> tuple[str, str]:
    if "/" not in repository:
        raise ValueError(f"repository must be in format 'owner/repo', got: {repository}")

    parts = repository.split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"repository must be in format 'owner/repo', got: {repository}")

    return parts[0], parts[1]


def parse_arguments() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_args")
    try:
        return Args.model_validate_json(parser.parse_args().json_args)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse input as JSON string: {e}")


def parse_docker_image(image: str) -> tuple[str, str, str]:
    if ':' in image:
        image_without_tag, tag = image.rsplit(':', 1)
    else:
        image_without_tag = image
        tag = ""

    if '/' not in image_without_tag:
        raise ValueError(f"Docker image must include registry (format: REGISTRY/IMAGE[:TAG]), got: {image}")

    registry, image_name = image_without_tag.split('/', 1)
    return registry, image_name, tag


def parse_env_vars(env_vars_str: str) -> dict[str, DynamicParameter]:
    env_dict: dict[str, DynamicParameter] = {}
    for line_num, pair in enumerate(env_vars_str.strip().split('\n'), 1):
        pair = pair.strip()
        if not pair:
            continue

        if '=' not in pair:
            raise ValueError(f"Invalid environment variable format at line {line_num}: '{pair}'. Expected KEY=VALUE format.")

        key, value = pair.split('=', 1)
        key = key.strip()

        if not key:
            raise ValueError(f"Empty environment variable key at line {line_num}")

        if not re.match(r'^[a-zA-Z0-9_-]+$', key):
            raise ValueError(f"Invalid environment variable key at line {line_num}: '{key}'. Keys must contain only alphanumeric characters, underscores, and hyphens.")

        if key in env_dict:
            raise ValueError(f"Duplicate environment variable key at line {line_num}: '{key}'")

        env_dict[key] = DynamicParameter(type="inline", value=value.strip())

    return env_dict


def parse_entrypoint_args(args_str: str) -> list[DynamicParameter]:
    args_list: list[DynamicParameter] = []
    for _, arg in enumerate(args_str.strip().split('\n'), 1):
        arg = arg.strip()
        if not arg:
            continue
        args_list.append(DynamicParameter(type="inline", value=arg))
    return args_list


def main():
    try:
        args = parse_arguments()
        payload = build_payload(args)
        print(payload.model_dump_json(indent=2))
        print("Successfully built request payload", file=sys.stderr)
    except Exception as e:
        print(f"Got error while creating request payload: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
