from typing import Annotated, Unpack

import click

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class DingoDBTypedDict(CommonTypedDict):
    addrs: Annotated[
        str,
        click.option("--addrs", type=str, help="Comma-separated DingoDB coordinator endpoints", required=True),
    ]
    index_name: Annotated[
        str,
        click.option(
            "--index-name",
            type=str,
            help="Index name to create/use",
            default="vectordb_bench_index",
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(DingoDBTypedDict)
def DingoDB(**parameters: Unpack[DingoDBTypedDict]):
    from .config import DingoDBCaseConfig, DingoDBConfig

    run(
        db=DB.DingoDB,
        db_config=DingoDBConfig(
            db_label=parameters["db_label"],
            addrs=parameters["addrs"],
            index_name=parameters["index_name"],
        ),
        db_case_config=DingoDBCaseConfig(),
        **parameters,
    )

