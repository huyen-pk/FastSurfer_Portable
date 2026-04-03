// This test suite integrates with test-containers for environment isolation.
use crate::inference::entities::InputVolume;
use crate::inference::pipeline::preprocess::{
    InferencePlane, batched_tensor_shape, load_input_volume, oriented_to_xyz,
    prepare_plane_input_for_slice, prepare_single_plane_input, scale_factor,
    slice_thickness_from_num_channels, thick_slice_channel_count,
    transformed_volume_shape, transformed_zoom,
};
use nifti::NiftiHeader;

fn build_indexed_volume() -> InputVolume {
    let shape_xyz = [2usize, 2usize, 3usize];
    let mut data_xyz = Vec::new();

    for x in 0..shape_xyz[0] {
        for y in 0..shape_xyz[1] {
            for z in 0..shape_xyz[2] {
                let value =
                    f32::from(u16::try_from((100 * x) + (10 * y) + z).expect(
                        "synthetic test coordinate should fit into u16",
                    ));
                data_xyz.push(value);
            }
        }
    }

    InputVolume {
        data_xyz,
        shape_xyz,
        zoom_xyz: [1.0, 2.0, 4.0],
        header: NiftiHeader::default(),
    }
}

fn build_volume_with_clamped_value() -> InputVolume {
    let mut volume = build_indexed_volume();
    volume.data_xyz[0] = 510.0;
    volume
}

fn normalized(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| (value / 255.0).clamp(0.0, 1.0))
        .collect::<Vec<f32>>()
}

fn assert_f32_slice_eq(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "tensor lengths should match");

    for (index, (left, right)) in actual.iter().zip(expected.iter()).enumerate()
    {
        assert!(
            (left - right).abs() <= 1e-6,
            "tensor mismatch at index {index}: actual={left} expected={right}"
        );
    }
}

fn assert_f32_eq(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 1e-6,
        "float mismatch: actual={actual} expected={expected}"
    );
}

fn assert_f32_array_eq<const N: usize>(actual: [f32; N], expected: [f32; N]) {
    assert_f32_slice_eq(&actual, &expected);
}

#[test]
fn transformed_volume_shape_should_match_plane_axis_order() {
    let shape = [11usize, 13usize, 17usize];

    assert_eq!(
        transformed_volume_shape(shape, InferencePlane::Coronal),
        [11, 13, 17]
    );
    assert_eq!(
        transformed_volume_shape(shape, InferencePlane::Axial),
        [17, 11, 13]
    );
    assert_eq!(
        transformed_volume_shape(shape, InferencePlane::Sagittal),
        [17, 13, 11]
    );
}

#[test]
fn transformed_zoom_and_scale_factor_should_match_plane_axis_order() {
    let zoom_xyz = [0.8f32, 1.1f32, 1.4f32];

    assert_f32_array_eq(
        transformed_zoom(zoom_xyz, InferencePlane::Coronal),
        [0.8, 1.1],
    );
    assert_f32_array_eq(
        transformed_zoom(zoom_xyz, InferencePlane::Axial),
        [1.4, 0.8],
    );
    assert_f32_array_eq(
        transformed_zoom(zoom_xyz, InferencePlane::Sagittal),
        [1.4, 1.1],
    );

    assert_f32_array_eq(scale_factor(2.8, [1.4, 0.8]), [2.0, 3.5]);
}

#[test]
fn oriented_to_xyz_should_map_plane_coordinates_to_volume_axes() {
    assert_eq!(oriented_to_xyz(InferencePlane::Coronal, 4, 5, 6), (4, 5, 6));
    assert_eq!(oriented_to_xyz(InferencePlane::Axial, 4, 5, 6), (5, 6, 4));
    assert_eq!(
        oriented_to_xyz(InferencePlane::Sagittal, 4, 5, 6),
        (6, 5, 4)
    );
}

#[test]
fn slice_thickness_helpers_should_derive_channel_and_batch_shapes() {
    let slice_thickness = slice_thickness_from_num_channels(7);

    assert_eq!(slice_thickness, 3);
    assert_eq!(thick_slice_channel_count(slice_thickness), 7);
    assert_eq!(
        batched_tensor_shape([17, 13, 11], slice_thickness),
        [11, 7, 17, 13]
    );
}

#[test]
fn prepare_plane_input_for_slice_should_build_expected_coronal_tensor() {
    let volume = build_indexed_volume();
    let prepared = prepare_plane_input_for_slice(
        &volume,
        InferencePlane::Coronal,
        3,
        2.0,
        1,
    )
    .expect("coronal plane input should prepare successfully");

    let expected = normalized(&[
        0.0, 10.0, 100.0, 110.0, 1.0, 11.0, 101.0, 111.0, 2.0, 12.0, 102.0,
        112.0,
    ]);

    assert_eq!(prepared.tensor_shape, [1, 3, 2, 2]);
    assert_f32_array_eq(prepared.scale_factor, [2.0, 1.0]);
    assert_eq!(prepared.plane, InferencePlane::Coronal);
    assert_eq!(prepared.slice_index, 1);
    assert_f32_slice_eq(&prepared.tensor_data, &expected);
}

#[test]
fn prepare_plane_input_for_slice_should_edge_pad_boundary_slices_and_clamp_values()
 {
    let volume = build_volume_with_clamped_value();
    let prepared = prepare_plane_input_for_slice(
        &volume,
        InferencePlane::Coronal,
        3,
        1.0,
        0,
    )
    .expect("boundary coronal plane input should prepare successfully");

    let first_channel = &prepared.tensor_data[0..4];
    let second_channel = &prepared.tensor_data[4..8];
    let third_channel = &prepared.tensor_data[8..12];
    let expected_edge = normalized(&[510.0, 10.0, 100.0, 110.0]);
    let expected_next = normalized(&[1.0, 11.0, 101.0, 111.0]);

    assert_f32_slice_eq(first_channel, &expected_edge);
    assert_f32_slice_eq(second_channel, &expected_edge);
    assert_f32_slice_eq(third_channel, &expected_next);
    assert_f32_eq(first_channel[0], 1.0);
    assert_f32_eq(second_channel[0], 1.0);
}

#[test]
fn prepare_plane_input_for_slice_should_build_expected_axial_tensor() {
    let volume = build_indexed_volume();
    let prepared = prepare_plane_input_for_slice(
        &volume,
        InferencePlane::Axial,
        1,
        4.0,
        1,
    )
    .expect("axial plane input should prepare successfully");

    let expected = normalized(&[10.0, 110.0, 11.0, 111.0, 12.0, 112.0]);

    assert_eq!(prepared.tensor_shape, [1, 1, 3, 2]);
    assert_f32_array_eq(prepared.scale_factor, [1.0, 4.0]);
    assert_eq!(prepared.plane, InferencePlane::Axial);
    assert_eq!(prepared.slice_index, 1);
    assert_f32_slice_eq(&prepared.tensor_data, &expected);
}

#[test]
fn prepare_single_plane_input_should_use_center_slice_for_requested_plane() {
    let volume = build_indexed_volume();
    let expected = prepare_plane_input_for_slice(
        &volume,
        InferencePlane::Sagittal,
        1,
        1.0,
        1,
    )
    .expect("explicit sagittal center slice should prepare successfully");

    let actual =
        prepare_single_plane_input(&volume, InferencePlane::Sagittal, 1, 1.0)
            .expect("single-plane helper should choose sagittal center slice");

    assert_eq!(actual.tensor_shape, expected.tensor_shape);
    assert_f32_array_eq(actual.scale_factor, expected.scale_factor);
    assert_eq!(actual.plane, expected.plane);
    assert_eq!(actual.slice_index, expected.slice_index);
    assert_f32_slice_eq(&actual.tensor_data, &expected.tensor_data);
}

#[test]
fn prepare_plane_input_for_slice_should_return_error_for_out_of_range_slice() {
    let volume = build_indexed_volume();
    let result = prepare_plane_input_for_slice(
        &volume,
        InferencePlane::Sagittal,
        1,
        1.0,
        2,
    );

    let Err(error) = result else {
        panic!("sagittal slice index outside transformed volume should fail");
    };

    assert_eq!(
        error,
        "Requested slice index 2 out of range for plane sagittal with 2 slices"
    );
}

#[test]
fn load_input_volume_should_reject_unsupported_input_extension() {
    let result = load_input_volume("/tmp/native-input.dcm");
    let Err(error) = result else {
        panic!("unsupported extension should fail before I/O");
    };

    assert_eq!(
        error,
        "Native Rust inference supports .nii/.nii.gz and .mgz/.mgh input, got: /tmp/native-input.dcm"
    );
}
