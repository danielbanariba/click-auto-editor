#!/usr/bin/env python3
"""
Test básico del efecto VHS

Verifica que todos los componentes funcionen correctamente.
"""

import sys
import os

# Añadir directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2


def create_test_frame(width=640, height=480):
    """Crea un frame de prueba con formas coloridas"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = [100, 150, 200]  # Color de fondo BGR

    # Rectángulo azul
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), -1)

    # Círculo verde
    cv2.circle(frame, (450, 240), 80, (0, 255, 0), -1)

    # Texto
    cv2.putText(frame, "VHS TEST", (150, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    return frame


def test_color_processor():
    """Test de conversiones de color RGB ↔ YIQ"""
    print("Testing ColorProcessor...")

    from vhs_effect.color_utils import ColorProcessor

    processor = ColorProcessor()

    # Test RGB a YIQ y vuelta
    rgb_test = np.array([[[1.0, 0.0, 0.0],    # Rojo
                          [0.0, 1.0, 0.0],    # Verde
                          [0.0, 0.0, 1.0]]])  # Azul

    yiq = processor.rgb_to_yiq(rgb_test)
    rgb_back = processor.yiq_to_rgb(yiq)

    # Verificar que la conversión es reversible (con tolerancia)
    assert np.allclose(rgb_test, rgb_back, atol=0.01), "RGB ↔ YIQ conversion failed"

    print("  ✓ RGB ↔ YIQ conversion OK")


def test_color_bleeding():
    """Test de color bleeding"""
    print("Testing color bleeding...")

    from vhs_effect.color_utils import ColorProcessor

    processor = ColorProcessor()
    frame = create_test_frame()

    result = processor.apply_color_bleeding(frame, intensity=0.7)

    assert result.shape == frame.shape, "Shape mismatch"
    assert result.dtype == np.uint8, "Dtype mismatch"
    assert not np.array_equal(result, frame), "Frame should be modified"

    print("  ✓ Color bleeding OK")


def test_chromatic_aberration():
    """Test de aberración cromática"""
    print("Testing chromatic aberration...")

    from vhs_effect.color_utils import apply_chromatic_aberration

    frame = create_test_frame()
    result = apply_chromatic_aberration(frame, intensity=0.7)

    assert result.shape == frame.shape, "Shape mismatch"
    assert result.dtype == np.uint8, "Dtype mismatch"

    print("  ✓ Chromatic aberration OK")


def test_vhs_effect_complete():
    """Test del procesador VHS completo"""
    print("Testing VHSEffect (complete)...")

    from vhs_effect import VHSEffect

    frame = create_test_frame()
    vhs = VHSEffect(intensity=0.7)

    result = vhs.process_frame(frame, frame_time=1.5)

    assert result.shape == frame.shape, "Shape mismatch"
    assert result.dtype == np.uint8, "Dtype mismatch"
    assert not np.array_equal(result, frame), "Frame should be modified"

    print("  ✓ VHSEffect complete OK")


def test_position_jitter():
    """Test de position jitter"""
    print("Testing PositionJitter...")

    from vhs_effect.vhs_processor import PositionJitter

    jitter = PositionJitter()

    # Obtener varios offsets
    offsets = []
    for i in range(100):
        offset = jitter.get_offset(i / 30.0, 30.0, 0.7)
        offsets.append(offset)

    # Verificar que hay variación (no todos iguales)
    unique_offsets = len(set(offsets))
    assert unique_offsets > 1, "Jitter should vary"

    print(f"  ✓ PositionJitter OK ({unique_offsets} unique offsets)")


def test_displacement_map():
    """Test de displacement maps"""
    print("Testing VHSDisplacementMap...")

    from vhs_effect import VHSDisplacementMap

    disp = VHSDisplacementMap(intensity=0.5)

    # Crear displacement map
    H, W = 480, 640
    disp_map = disp.create_displacement_map(H, W, time_t=1.0)

    assert disp_map.shape == (H, W), "Shape mismatch"
    assert disp_map.min() >= 0.0, "Values should be >= 0"
    assert disp_map.max() <= 1.0, "Values should be <= 1"

    # Aplicar displacement
    frame = create_test_frame(W, H)
    result = disp.apply_displacement(frame, disp_map)

    assert result.shape == frame.shape, "Shape mismatch after displacement"

    print("  ✓ VHSDisplacementMap OK")


def test_overlays():
    """Test de overlays VCR"""
    print("Testing VHSTextOverlay...")

    from vhs_effect import VHSTextOverlay

    frame = create_test_frame()

    # Test timecode
    result = VHSTextOverlay.add_timecode(frame, 125.5)
    assert result.shape == frame.shape, "Timecode shape mismatch"

    # Test REC indicator
    result = VHSTextOverlay.add_rec_indicator(frame)
    assert result.shape == frame.shape, "REC indicator shape mismatch"

    # Test PLAY indicator
    result = VHSTextOverlay.add_play_indicator(frame)
    assert result.shape == frame.shape, "PLAY indicator shape mismatch"

    # Test full overlay
    result = VHSTextOverlay.add_full_vcr_overlay(frame, 125.5, frame_count=10)
    assert result.shape == frame.shape, "Full overlay shape mismatch"

    print("  ✓ VHSTextOverlay OK")


def test_save_result():
    """Guarda resultado visual para inspección manual"""
    print("Saving test result...")

    from vhs_effect import VHSEffect, VHSTextOverlay

    frame = create_test_frame()
    vhs = VHSEffect(intensity=0.7)

    result = vhs.process_frame(frame, frame_time=1.5)
    result = VHSTextOverlay.add_full_vcr_overlay(result, 125.5, frame_count=10)

    output_path = "test_output.jpg"
    cv2.imwrite(output_path, result)

    assert os.path.exists(output_path), "Output file not created"
    assert os.path.getsize(output_path) > 0, "Output file is empty"

    print(f"  ✓ Saved: {output_path}")


def main():
    print("=" * 50)
    print("VHS Effect - Test Suite")
    print("=" * 50)
    print()

    try:
        test_color_processor()
        test_color_bleeding()
        test_chromatic_aberration()
        test_vhs_effect_complete()
        test_position_jitter()
        test_displacement_map()
        test_overlays()
        test_save_result()

        print()
        print("=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)

    except Exception as e:
        print()
        print("=" * 50)
        print(f"TEST FAILED: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
