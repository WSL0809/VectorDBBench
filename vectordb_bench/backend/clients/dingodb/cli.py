from typing import Annotated, Unpack
import json

import click

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    click_arg_split,
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
    enable_scalar_speed_up_with_document: Annotated[
        bool,
        click.option(
            "--enable-document-speedup/--disable-document-speedup",
            "enable_scalar_speed_up_with_document",
            is_flag=True,
            default=True,
            show_default=True,
            help="Enable document-accelerated scalar filtering (query_string path)",
        ),
    ]
    # Accept batch/yaml alias --enable-scalar-speed-up-with-document as a flag
    enable_scalar_speed_up_with_document_alias: Annotated[
        bool,
        click.option(
            "--enable-scalar-speed-up-with-document",
            "enable_scalar_speed_up_with_document",
            is_flag=True,
            default=None,
            help="Alias for --enable-document-speedup",
        ),
    ]
    scalar_speedup_operand: Annotated[
        list[int] | None,
        click.option(
            "--scalar-speedup-operand",
            type=str,
            default=None,
            help=(
                "Partition operands for document speedup. Accepts '5,10,15,20' or a JSON/list string like "
                "'[5, 10, 15, 20]'"
            ),
            callback=lambda ctx, param, value: (
                None
                if value is None
                else (
                    json.loads(value)
                    if (value.strip().startswith("[") and value.strip().endswith("]"))
                    else list(map(int, click_arg_split(ctx, param, value)))
                )
            ),
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
            enable_scalar_speed_up_with_document=parameters["enable_scalar_speed_up_with_document"],
            scalar_speedup_operand=parameters["scalar_speedup_operand"],
        ),
        db_case_config=DingoDBCaseConfig(),
        **parameters,
    )
