import unittest

from fast_cls.gpu_monitor import DeviceIdentity, format_device_group, _nvml_device


class TestGpuMonitor(unittest.TestCase):
    def test_device_confirmation_reports_mapping_failure(self):
        identity = DeviceIdentity(0, 0, 'cuda', 0, 'Tesla V100-SXM3-32GB',
                                  mapping_error='CUDA Runtime library unavailable')
        report = format_device_group([identity])
        self.assertIn('mapping_error=CUDA Runtime library unavailable', report)

    def test_cuda_device_maps_to_same_nvml_pci_device(self):
        import torch
        import pynvml

        if not torch.cuda.is_available():
            self.skipTest('CUDA is unavailable')
        pynvml.nvmlInit()
        try:
            for cuda_index in range(torch.cuda.device_count()):
                with self.subTest(cuda_index=cuda_index):
                    handle, _, pci = _nvml_device(cuda_index)
                    nvml_pci = pynvml.nvmlDeviceGetPciInfo(handle).busId
                    if isinstance(nvml_pci, bytes):
                        nvml_pci = nvml_pci.decode()
                    self.assertEqual(pci.lower(), nvml_pci.lower())
        finally:
            pynvml.nvmlShutdown()


if __name__ == '__main__':
    unittest.main()
