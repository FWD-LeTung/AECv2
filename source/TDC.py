import torch
import torch.nn as nn
import torch.nn.functional as F

class SubbandTDC(nn.Module):
    def __init__(self, frame_len=512, hop_len=256, num_subbands=4):
        super().__init__()
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.num_subbands = num_subbands

    def forward(self, mic, ref):
        """
        Input: mic, ref (Batch, Time)
        Output: aligned_ref (Batch, Time), delay (int)
        """
        # 1. Chuyển sang miền tần số (STFT đóng vai trò là Subband decomposition)
        # Ref: "TDC ... performed on the subband features" [cite: 967]
        mic_stft = torch.stft(mic, n_fft=self.frame_len, hop_length=self.hop_len, 
                              return_complex=True, window=torch.hann_window(self.frame_len).to(mic.device))
        ref_stft = torch.stft(ref, n_fft=self.frame_len, hop_length=self.hop_len, 
                              return_complex=True, window=torch.hann_window(self.frame_len).to(ref.device))
        
        # (Batch, Freq, Time)
        
        # 2. Tính GCC-PHAT trên miền tần số
        # Cross-correlation: R = X * conj(Y) / |X * conj(Y)|
        cc = ref_stft * torch.conj(mic_stft)
        cc_phat = cc / (torch.abs(cc) + 1e-8)
        
        # 3. Chia thành K subbands dọc theo trục tần số (Freq bins)
        # Ref: "estimating several time delays in each subband" 
        n_freq = cc_phat.shape[1]
        band_size = n_freq // self.num_subbands
        delays = []
        
        for i in range(self.num_subbands):
            start = i * band_size
            end = (i + 1) * band_size if i < self.num_subbands - 1 else n_freq
            
            # Lấy cross-correlation của subband đó
            subband_cc = cc_phat[:, start:end, :]
            
            # IFFT để về miền thời gian (lag domain) của subband đó
            # Cần zero-pad để khôi phục độ phân giải thời gian nếu cần, 
            # nhưng ở đây ta tính trung bình GCC trên trục Time frames trước
            avg_cc = torch.mean(subband_cc, dim=1) # Average over freq in band
            
            # IFFT trục thời gian để tìm lag (gcc_time)
            # Lưu ý: Thực hiện GCC chuẩn thường cần IFFT trên toàn bộ phổ.
            # Để đơn giản và hiệu quả trong PyTorch:
            # Ta thực hiện GCC-PHAT global nhưng mask các tần số khác đi
            mask = torch.zeros_like(cc_phat)
            mask[:, start:end, :] = 1.0
            masked_cc = cc_phat * mask
            
            # Sum over frequency -> IFFT
            gcc_t = torch.fft.irfft(torch.sum(masked_cc, dim=1), n=self.frame_len)
            
            # Tìm max lag
            # Dịch chuyển để lag 0 nằm giữa
            gcc_t = torch.roll(gcc_t, shifts=self.frame_len//2, dims=-1)
            est_delay = torch.argmax(gcc_t, dim=-1) - self.frame_len//2
            delays.append(est_delay)

        # 4. Voting Method: Chọn delay xuất hiện nhiều nhất
        # Ref: "final time delay is determined using a simple voting method" 
        delays_tensor = torch.stack(delays, dim=0) # (Subbands, Batch)
        final_delays = []
        
        for b in range(mic.shape[0]):
            vals, counts = torch.unique(delays_tensor[:, b], return_counts=True)
            best_idx = torch.argmax(counts)
            final_delays.append(vals[best_idx].item())
            
        final_delay = int(sum(final_delays) / len(final_delays)) # Lấy trung bình batch (hoặc xử lý từng sample)
        
        # 5. Căn chỉnh tín hiệu Ref
        # Shift ref signal by delay
        if final_delay > 0:
            aligned_ref = torch.nn.functional.pad(ref, (0, final_delay))[:, :-final_delay]
        elif final_delay < 0:
            aligned_ref = torch.nn.functional.pad(ref, (-final_delay, 0))[:, -final_delay:]
        else:
            aligned_ref = ref
            
        return aligned_ref, final_delay