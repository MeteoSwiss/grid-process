# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DataModelOptions = Annotated[list[str], Field(..., min_length=1)]


class CacheSizeSettings(BaseModel):
    nn_lookup: int
    nn_lookup_regular: int
    reproject_poi: int
    reproject_grid: int


class GridSettings(BaseModel):
    unstructured_dim: DataModelOptions
    y: DataModelOptions
    x: DataModelOptions
    surf_alt: DataModelOptions


class PoiSettings(BaseModel):
    dim: str
    coord_from: str
    alt_from: DataModelOptions


class DataModelSettings(BaseModel):
    grid: GridSettings
    poi: PoiSettings
    validtime: DataModelOptions
    leadtime: DataModelOptions
    reftime: DataModelOptions
    ens_dim: DataModelOptions


class LibrarySettings(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file=os.path.join(os.path.dirname(__file__), "grid_process.yaml"),
        env_prefix="GRID_PROCESS__",
        env_nested_delimiter="__",
    )
    cache_size: CacheSizeSettings
    data_model: DataModelSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (env_settings, YamlConfigSettingsSource(settings_cls))
