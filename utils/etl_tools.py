import json
from pathlib import Path

import pandas as pd
import requests

from models.schema import (
    CastColumnsOperation,
    DropColumnsOperation,
    DropDuplicatesOperation,
    FillMissingOperation,
    FilterRowsOperation,
    GroupByAggregateOperation,
    RenameColumnsOperation,
    SelectColumnsOperation,
    SortValuesOperation,
    StringTransformOperation,
    TransformPlan,
)


class ETLTools:
    """
    Deterministic ETL operations used by the ETL agent.

    The class intentionally does NOT execute arbitrary Python code.
    All transformations must use explicitly supported operations.
    """

    SUPPORTED_FORMATS = {
        "csv",
        "json",
        "parquet",
    }

    def __init__(self):

        self.project_root = Path(
            __file__
        ).resolve().parents[1]

        self.data_root = (
            self.project_root / "data"
        ).resolve()


    # ============================================================
    # PATH SAFETY
    # ============================================================

    def _resolve_data_path(
        self,
        path: str,
        must_exist: bool = False,
    ) -> Path:
        """
        Resolve a path while ensuring that it stays inside
        the project's data/ directory.

        This prevents the ETL agent from accessing files such as:

        .env
        source code
        SSH keys
        arbitrary system files
        """

        candidate = Path(path)

        if not candidate.is_absolute():
            candidate = (
                self.project_root / candidate
            )

        candidate = candidate.resolve()

        try:
            candidate.relative_to(
                self.data_root
            )

        except ValueError as exc:
            raise ValueError(
                "ETL file operations are restricted "
                "to the project's data directory."
            ) from exc

        if must_exist and not candidate.exists():
            raise FileNotFoundError(
                f"File does not exist: {candidate}"
            )

        return candidate


    # ============================================================
    # FORMAT VALIDATION
    # ============================================================

    def _validate_format(
        self,
        file_format: str,
    ) -> str:

        normalized = (
            file_format
            .lower()
            .strip()
            .lstrip(".")
        )

        if normalized not in self.SUPPORTED_FORMATS:

            raise ValueError(
                f"Unsupported format: {file_format}. "
                f"Supported formats: "
                f"{sorted(self.SUPPORTED_FORMATS)}"
            )

        return normalized


    # ============================================================
    # DATAFRAME IO
    # ============================================================

    def _load_dataframe(
        self,
        file_path: str,
    ) -> pd.DataFrame:

        path = self._resolve_data_path(
            file_path,
            must_exist=True,
        )

        extension = (
            path.suffix
            .lower()
        )

        if extension == ".csv":

            return pd.read_csv(
                path
            )

        if extension == ".json":

            try:
                return pd.read_json(
                    path,
                    lines=True,
                )

            except ValueError:

                return pd.read_json(
                    path
                )

        if extension == ".parquet":

            return pd.read_parquet(
                path
            )

        raise ValueError(
            f"Unsupported input format: "
            f"{extension}"
        )


    def _save_dataframe(
        self,
        dataframe: pd.DataFrame,
        file_path: Path,
        file_format: str,
    ) -> None:

        file_format = self._validate_format(
            file_format
        )

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if file_format == "csv":

            dataframe.to_csv(
                file_path,
                index=False,
            )

            return

        if file_format == "json":

            dataframe.to_json(
                file_path,
                orient="records",
                lines=True,
            )

            return

        if file_format == "parquet":

            dataframe.to_parquet(
                file_path,
                index=False,
            )

            return


    # ============================================================
    # EXTRACTION
    # ============================================================

    def extract_load(
        self,
        url: str,
        output_folder: str,
        format: str,
    ) -> str:
        """
        Extract JSON data from an API and store it locally.
        """

        file_format = self._validate_format(
            format
        )

        output_directory = (
            self._resolve_data_path(
                output_folder
            )
        )

        response = requests.get(
            url,
            timeout=30,
        )

        response.raise_for_status()

        payload = response.json()

        # Support common API response structures.
        if (
            isinstance(payload, dict)
            and isinstance(
                payload.get("results"),
                list,
            )
        ):

            records = payload["results"]

        elif isinstance(
            payload,
            list,
        ):

            records = payload

        elif isinstance(
            payload,
            dict,
        ):

            records = [payload]

        else:

            raise ValueError(
                "API returned an unsupported "
                "JSON structure."
            )

        dataframe = pd.json_normalize(
            records
        )

        output_file = (
            output_directory
            / f"extracted_data.{file_format}"
        )

        self._save_dataframe(
            dataframe=dataframe,
            file_path=output_file,
            file_format=file_format,
        )

        return (
            "Data successfully extracted.\n"
            f"Rows: {len(dataframe)}\n"
            f"Columns: {len(dataframe.columns)}\n"
            f"Output: {output_file}"
        )


    # ============================================================
    # DATASET CONTEXT
    # ============================================================

    def get_dataset_context(
        self,
        file_path: str,
    ) -> str:
        """
        Return useful metadata for the transformation planner.

        We provide metadata and a small sample rather than giving
        the LLM the entire dataset.
        """

        dataframe = self._load_dataframe(
            file_path
        )

        sample_json = dataframe.head(
            5
        ).to_json(
            orient="records",
            date_format="iso",
        )

        context = {
            "row_count": len(
                dataframe
            ),
            "columns": list(
                dataframe.columns
            ),
            "dtypes": {
                column: str(dtype)
                for column, dtype
                in dataframe.dtypes.items()
            },
            "null_counts": {
                column: int(count)
                for column, count
                in dataframe.isnull().sum().items()
            },
            "sample_rows": json.loads(
                sample_json
            ),
        }

        return json.dumps(
            context,
            indent=2,
            default=str,
        )


    # ============================================================
    # COLUMN VALIDATION
    # ============================================================

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> None:

        missing = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing:

            raise ValueError(
                "Transformation references "
                f"missing columns: {missing}"
            )


    # ============================================================
    # FILTERING
    # ============================================================

    def _apply_filter(
        self,
        dataframe: pd.DataFrame,
        operation: FilterRowsOperation,
    ) -> pd.DataFrame:

        self._validate_columns(
            dataframe,
            [operation.column],
        )

        series = dataframe[
            operation.column
        ]

        operator = (
            operation.operator
        )

        value = (
            operation.value
        )

        if operator == "is_null":

            mask = series.isna()

        elif operator == "not_null":

            mask = series.notna()

        elif operator == "contains":

            if not isinstance(
                value,
                str,
            ):
                raise ValueError(
                    "'contains' requires "
                    "a string value."
                )

            mask = (
                series
                .astype("string")
                .str.contains(
                    value,
                    case=operation.case_sensitive,
                    regex=False,
                    na=False,
                )
            )

        elif operator in {
            "in",
            "not_in",
        }:

            if not isinstance(
                value,
                list,
            ):
                raise ValueError(
                    f"'{operator}' requires "
                    "a list value."
                )

            mask = series.isin(
                value
            )

            if operator == "not_in":

                mask = ~mask

        elif operator == "eq":

            mask = (
                series == value
            )

        elif operator == "ne":

            mask = (
                series != value
            )

        elif operator == "gt":

            mask = (
                series > value
            )

        elif operator == "gte":

            mask = (
                series >= value
            )

        elif operator == "lt":

            mask = (
                series < value
            )

        elif operator == "lte":

            mask = (
                series <= value
            )

        else:

            raise ValueError(
                f"Unsupported filter "
                f"operator: {operator}"
            )

        return dataframe.loc[
            mask
        ].copy()


    # ============================================================
    # TYPE CASTING
    # ============================================================

    def _cast_columns(
        self,
        dataframe: pd.DataFrame,
        operation: CastColumnsOperation,
    ) -> pd.DataFrame:

        self._validate_columns(
            dataframe,
            list(
                operation.dtypes.keys()
            ),
        )

        result = dataframe.copy()

        for column, target_type in (
            operation.dtypes.items()
        ):

            if target_type == "string":

                result[column] = (
                    result[column]
                    .astype("string")
                )

            elif target_type == "integer":

                result[column] = (
                    pd.to_numeric(
                        result[column],
                        errors="raise",
                    )
                    .astype("Int64")
                )

            elif target_type == "float":

                result[column] = (
                    pd.to_numeric(
                        result[column],
                        errors="raise",
                    )
                    .astype(float)
                )

            elif target_type == "datetime":

                result[column] = (
                    pd.to_datetime(
                        result[column],
                        errors="raise",
                    )
                )

            elif target_type == "category":

                result[column] = (
                    result[column]
                    .astype("category")
                )

            elif target_type == "boolean":

                if pd.api.types.is_bool_dtype(
                    result[column]
                ):

                    result[column] = (
                        result[column]
                        .astype("boolean")
                    )

                else:

                    normalized = (
                        result[column]
                        .astype("string")
                        .str.strip()
                        .str.lower()
                    )

                    mapping = {
                        "true": True,
                        "1": True,
                        "yes": True,
                        "false": False,
                        "0": False,
                        "no": False,
                    }

                    unknown = (
                        normalized
                        .dropna()
                        .loc[
                            ~normalized
                            .dropna()
                            .isin(mapping)
                        ]
                        .unique()
                    )

                    if len(
                        unknown
                    ) > 0:

                        raise ValueError(
                            "Cannot safely convert "
                            f"{column} to boolean. "
                            f"Unknown values: "
                            f"{list(unknown)}"
                        )

                    result[column] = (
                        normalized
                        .map(mapping)
                        .astype("boolean")
                    )

        return result


    # ============================================================
    # APPLY ONE OPERATION
    # ============================================================

    def _apply_operation(
        self,
        dataframe: pd.DataFrame,
        operation,
    ) -> pd.DataFrame:

        # --------------------------------------------------------
        # SELECT COLUMNS
        # --------------------------------------------------------

        if isinstance(
            operation,
            SelectColumnsOperation,
        ):

            self._validate_columns(
                dataframe,
                operation.columns,
            )

            return dataframe.loc[
                :,
                operation.columns,
            ].copy()


        # --------------------------------------------------------
        # DROP COLUMNS
        # --------------------------------------------------------

        if isinstance(
            operation,
            DropColumnsOperation,
        ):

            self._validate_columns(
                dataframe,
                operation.columns,
            )

            return dataframe.drop(
                columns=operation.columns
            )


        # --------------------------------------------------------
        # RENAME COLUMNS
        # --------------------------------------------------------

        if isinstance(
            operation,
            RenameColumnsOperation,
        ):

            self._validate_columns(
                dataframe,
                list(
                    operation.mapping.keys()
                ),
            )

            return dataframe.rename(
                columns=operation.mapping
            )


        # --------------------------------------------------------
        # FILTER ROWS
        # --------------------------------------------------------

        if isinstance(
            operation,
            FilterRowsOperation,
        ):

            return self._apply_filter(
                dataframe,
                operation,
            )


        # --------------------------------------------------------
        # DROP DUPLICATES
        # --------------------------------------------------------

        if isinstance(
            operation,
            DropDuplicatesOperation,
        ):

            if operation.subset:

                self._validate_columns(
                    dataframe,
                    operation.subset,
                )

            return dataframe.drop_duplicates(
                subset=operation.subset,
                keep=operation.keep,
            )


        # --------------------------------------------------------
        # SORT
        # --------------------------------------------------------

        if isinstance(
            operation,
            SortValuesOperation,
        ):

            self._validate_columns(
                dataframe,
                operation.columns,
            )

            return dataframe.sort_values(
                by=operation.columns,
                ascending=operation.ascending,
                kind="stable",
            )


        # --------------------------------------------------------
        # FILL MISSING VALUES
        # --------------------------------------------------------

        if isinstance(
            operation,
            FillMissingOperation,
        ):

            self._validate_columns(
                dataframe,
                list(
                    operation.values.keys()
                ),
            )

            return dataframe.fillna(
                value=operation.values
            )


        # --------------------------------------------------------
        # CAST TYPES
        # --------------------------------------------------------

        if isinstance(
            operation,
            CastColumnsOperation,
        ):

            return self._cast_columns(
                dataframe,
                operation,
            )


        # --------------------------------------------------------
        # STRING TRANSFORMATIONS
        # --------------------------------------------------------

        if isinstance(
            operation,
            StringTransformOperation,
        ):

            self._validate_columns(
                dataframe,
                operation.columns,
            )

            result = dataframe.copy()

            for column in (
                operation.columns
            ):

                values = (
                    result[column]
                    .astype("string")
                )

                if operation.action == "strip":

                    values = (
                        values.str.strip()
                    )

                elif operation.action == "lower":

                    values = (
                        values.str.lower()
                    )

                elif operation.action == "upper":

                    values = (
                        values.str.upper()
                    )

                result[column] = values

            return result


        # --------------------------------------------------------
        # GROUP BY / AGGREGATE
        # --------------------------------------------------------

        if isinstance(
            operation,
            GroupByAggregateOperation,
        ):

            self._validate_columns(
                dataframe,
                operation.group_by,
            )

            aggregation_columns = [
                item.column
                for item
                in operation.aggregations
            ]

            self._validate_columns(
                dataframe,
                aggregation_columns,
            )

            named_aggregations = {}

            for item in (
                operation.aggregations
            ):

                if (
                    item.alias
                    in named_aggregations
                ):
                    raise ValueError(
                        "Duplicate aggregation "
                        f"alias: {item.alias}"
                    )

                named_aggregations[
                    item.alias
                ] = pd.NamedAgg(
                    column=item.column,
                    aggfunc=item.function,
                )

            return (
                dataframe
                .groupby(
                    operation.group_by,
                    dropna=False,
                )
                .agg(
                    **named_aggregations
                )
                .reset_index()
            )


        raise ValueError(
            "Unsupported transformation "
            f"operation: {operation}"
        )


    # ============================================================
    # EXECUTE TRANSFORMATION PLAN
    # ============================================================

    def apply_transform_plan(
        self,
        dataframe: pd.DataFrame,
        plan: TransformPlan,
    ) -> pd.DataFrame:
        """
        Apply a validated sequence of deterministic operations.
        """

        result = dataframe.copy()

        for operation in (
            plan.operations
        ):

            result = self._apply_operation(
                result,
                operation,
            )

        return result


    # ============================================================
    # TRANSFORM + LOAD
    # ============================================================

    def transform_load(
        self,
        input_file_path: str,
        output_folder: str,
        output_format: str,
        plan: TransformPlan,
    ) -> str:

        file_format = (
            self._validate_format(
                output_format
            )
        )

        dataframe = (
            self._load_dataframe(
                input_file_path
            )
        )

        original_rows = len(
            dataframe
        )

        transformed = (
            self.apply_transform_plan(
                dataframe,
                plan,
            )
        )

        output_directory = (
            self._resolve_data_path(
                output_folder
            )
        )

        output_file = (
            output_directory
            / f"transformed_data.{file_format}"
        )

        self._save_dataframe(
            dataframe=transformed,
            file_path=output_file,
            file_format=file_format,
        )

        return (
            "Transformation completed successfully.\n"
            f"Input rows: {original_rows}\n"
            f"Output rows: {len(transformed)}\n"
            f"Output columns: "
            f"{list(transformed.columns)}\n"
            f"Output file: {output_file}\n"
            f"Plan summary: {plan.summary}"
        )