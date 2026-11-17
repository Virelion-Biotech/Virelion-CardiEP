import pytest
from pydantic import ValidationError

from cardiep.models import (
    ArtifactRef,
    EPDiseaseContext,
    EPParameterSet,
    EPSimulationRequest,
    EPSimulationResult,
)
from cardiep.service import CardiEPService, ReadinessError


def source():
    return [{"source_identifier":"PMID:123","source_type":"peer_reviewed","source_url":"https://example.org/source","source_sha256":"a"*64}]

def context(**kw):
    base={"disease_label":"PLN_cardiomyopathy","genotype_status":"curated","variants":[{"gene_symbol":"PLN","variant_nomenclature":"NM_002667.5:c.40_42del"}],"source_record_ids":["atlas-claim-1"],"source_provenance":source(),"phenotype_label":"PLN-associated cardiomyopathy phenotype","phenotype_label_source_record_ids":["phenotype-1"],"phenotype_label_source_provenance":source(),"electrotrace_bundle_sha256":["b"*64],"baseline_definition":{"kind":"pre_intervention"},"comparator_definition":{"kind":"vehicle_or_reference"},"uncertainty_reference":{"status":"not_characterized"},"calibration_cohort":{"cohort_id":"cohort-1"},"subject_split":{"subject_id":"S1","split_group_id":"S1"},"requested_mechanisms":["conduction"]}
    base.update(kw); return EPDiseaseContext(**base)

def request(ctx,values=None):
    return EPSimulationRequest(subject_id="S1",anatomy_ref=ArtifactRef(artifact_id="anat",kind="ep_geometry",uri="file:///tmp/a.json"),backend="numpy-eikonal-v1",parameters=EPParameterSet(values=values or {"fibre_speed":0.065},source="fixed"),disease_context=ctx)

def test_genotype_and_disease_context_do_not_modify_ep_parameters():
    ctx=context()
    req=request(ctx)
    assert req.parameters.values=={"fibre_speed":0.065}
    assert req.disease_context.variants[0]["gene_symbol"]=="PLN"

def test_genotype_cannot_be_inferred_from_ecg():
    with pytest.raises(ValidationError,match="never be inferred"):
        context(inferred_from_ecg=True)

def test_reported_genotype_requires_provenance():
    with pytest.raises(ValidationError,match="source record IDs and provenance"):
        context(genotype_status="reported",source_record_ids=[],source_provenance=[])

def test_disease_label_requires_independent_label_provenance():
    with pytest.raises(ValidationError,match="phenotype source record IDs"):
        context(phenotype_label_source_record_ids=[],phenotype_label_source_provenance=[])

def test_native_eikonal_rejects_intracellular_calcium_and_variant_ionic_mechanisms():
    with pytest.raises(ValidationError,match="does not support disease mechanisms"):
        request(context(requested_mechanisms=["intracellular_calcium_cycling"]))

def test_calibrated_mapping_must_match_backend_and_explicit_parameters():
    mapping={"mapping_id":"pln-map-v1","backend":"numpy-eikonal-v1","calibration_evidence_ids":["cal-1"],"source_bundle_sha256":["b"*64],"calibration_cohort":{"cohort_id":"C1"},"subject_split":{"split_group_id":"S1"},"parameters":{"fibre_speed":0.08}}
    with pytest.raises(ValidationError,match="differs from explicit calibrated disease mapping"):
        request(context(parameter_mapping=mapping),values={"fibre_speed":0.065})

def test_external_backend_must_declare_mechanism_capabilities():
    class Backend:
        name="external-unqualified"
    with pytest.raises(ReadinessError,match="must explicitly declare"):
        CardiEPService._validate_disease_mechanism_capabilities(Backend(),context(requested_mechanisms=["intracellular_calcium_cycling"]))

def test_native_backend_supported_mechanisms_are_limited():
    class Backend:
        name="numpy-eikonal-v1"
    CardiEPService._validate_disease_mechanism_capabilities(Backend(),context(requested_mechanisms=["conduction","scar_tissue_properties"]))
    with pytest.raises(ReadinessError,match="does not support"):
        CardiEPService._validate_disease_mechanism_capabilities(Backend(),context(requested_mechanisms=["proteostasis"]))

def test_disease_context_is_preserved_in_result_provenance():
    ctx=context()
    req=request(ctx)
    result=EPSimulationResult(subject_id="S1",backend="numpy-eikonal-v1",parameters=req.parameters)
    CardiEPService._preserve_disease_context(req,result)
    assert result.provenance["disease_context"]["disease_label"]=="PLN_cardiomyopathy"
    assert len(result.provenance["disease_context_sha256"])==64


def test_calibration_mapping_requires_matching_initial_parameters():
    from cardiep.models import EPCalibrationRequest, EPObservation

    mapping={"mapping_id":"pln-map-v1","backend":"numpy-eikonal-v1","calibration_evidence_ids":["cal-1"],"source_bundle_sha256":["b"*64],"calibration_cohort":{"cohort_id":"C1"},"subject_split":{"split_group_id":"S1"},"parameters":{"fibre_speed":0.08}}
    ctx=context(parameter_mapping=mapping)
    observation=EPObservation(observation_id="ecg1",kind="ecg",artifact=ArtifactRef(artifact_id="ecg",kind="electrotrace_ecg_calibration",uri="file:///tmp/ecg.json",sha256="b"*64))
    with pytest.raises(ValidationError,match="differs from explicit calibrated disease mapping"):
        EPCalibrationRequest(subject_id="S1",anatomy_ref=ArtifactRef(artifact_id="anat",kind="ep_geometry",uri="file:///tmp/a.json"),observations=[observation],backend="numpy-eikonal-v1",parameter_bounds={"fibre_speed":(0.05,0.15)},initial_parameters=EPParameterSet(values={"fibre_speed":0.065}),disease_context=ctx)


def test_explicit_calibrated_mapping_is_preserved_when_values_match():
    mapping = {
        "mapping_id": "pln-map-v1",
        "backend": "numpy-eikonal-v1",
        "calibration_evidence_ids": ["cal-1"],
        "source_bundle_sha256": ["b" * 64],
        "calibration_cohort": {"cohort_id": "C1"},
        "subject_split": {"split_group_id": "S1"},
        "parameters": {"fibre_speed": 0.08},
    }
    calibrated_request = request(
        context(parameter_mapping=mapping),
        values={"fibre_speed": 0.08},
    )
    assert calibrated_request.parameters.values["fibre_speed"] == 0.08
    assert (
        calibrated_request.disease_context.parameter_mapping["mapping_id"]
        == "pln-map-v1"
    )
