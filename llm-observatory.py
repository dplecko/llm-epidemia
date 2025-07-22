
import datasets
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from huggingface_hub import hf_hub_download

FILE_NAMES = [
    "acs", "brfss", "census", "edu", "fbi_arrests",
    "gss", "labor", "meps", "nhanes", "nsduh", "scf",
]

def _make_configs():
    return [
        datasets.BuilderConfig(
            name=fn,
            version=datasets.Version("1.0.0"),
            description=f"Rows from {fn}.parquet",
        )
        for fn in FILE_NAMES
    ]

class Observatory(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = _make_configs()
    DEFAULT_CONFIG_NAME = "acs"

    def _info(self):
        # 1) download the parquet for this config from HF
        parquet_path = hf_hub_download(
            repo_id="llm-observatory/llm-observatory",
            filename=f"data/clean/{self.config.name}.parquet",
            repo_type="dataset",
        )
        # 2) read its Arrow schema
        pa_schema = pq.read_schema(parquet_path)
        # 3) strip out any dictionary-encoded types
        cleaned_fields = []
        for field in pa_schema:
            if pa.types.is_dictionary(field.type):
                # use the underlying value type instead
                new_type = field.type.value_type
                cleaned_fields.append(
                    pa.field(field.name, new_type, nullable=field.nullable, metadata=field.metadata)
                )
            else:
                cleaned_fields.append(field)
        cleaned_schema = pa.schema(cleaned_fields, metadata=pa_schema.metadata)
        # 4) convert to HF Features
        features = datasets.Features.from_arrow_schema(cleaned_schema)
        return datasets.DatasetInfo(
            description="Tabular benchmarks from LLM-Observatory.",
            features=features,
        )

    def _split_generators(self, dl_manager):
        parquet_path = dl_manager.download(f"data/{self.config.name}.parquet")
        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={"path": parquet_path},
            )
        ]

    def _generate_examples(self, path):
        df = pd.read_parquet(path)
        # replace NaN with None so Arrow can handle missing values
        df = df.where(pd.notnull(df), None)
        # decode any bytes into str
        for col in df.select_dtypes(include=[object]).columns:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
            )
        for idx, record in enumerate(df.to_dict(orient="records")):
            yield idx, record

builder_cls = Observatory