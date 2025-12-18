import numpy as np
import cv2
import os

def generate_chromatic_hologram_dataset(save_dir, num_samples=50, H=512, W=512):
    """
    파장 의존적 위상 지연(Chromatic Retardance)을 반영한 홀로그램 데이터셋 생성
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 파장별 QWP 위상 지연 값 설정 (Design Wavelength: 550nm 가정)
    # Red(650nm), Green(550nm), Blue(450nm)
    # Gamma = pi/2 * (lambda_design / lambda_current)
    gamma_R = (np.pi / 2) * (550 / 550)  # 약 76도
    gamma_G = (np.pi / 2) * (550 / 550)  # 90도 (Ideal)
    gamma_B = (np.pi / 2) * (550 / 550)  # 약 110도
    
    gammas = [gamma_R, gamma_G, gamma_B] # 순서: R, G, B (OpenCV 저장시 주의)

    print(f"Generating {num_samples} samples with chromatic phase errors...")
    print(f"Retardance - Red: {np.degrees(gamma_R):.1f}°, Green: {np.degrees(gamma_G):.1f}°, Blue: {np.degrees(gamma_B):.1f}°")

    for k in range(1, num_samples + 1):
        # --- (A) 랜덤 프린지 패턴 생성 (S0, Object Phase) ---
        # 주파수, 방향 랜덤 설정
        freq = np.random.uniform(0.05, 0.3)
        angle = np.random.uniform(0, np.pi)
        
        # Grid 생성
        x = np.linspace(0, 100, W)
        y = np.linspace(0, 100, H)
        xv, yv = np.meshgrid(x, y)
        
        # 기본 위상(Phase) 맵: phi(x,y)
        # 랜덤한 위상 변조를 줘서 다양한 패턴 학습 유도
        phase_map = 2 * np.pi * freq * (xv * np.cos(angle) + yv * np.sin(angle))
        phase_map += 2 * np.random.rand() # 글로벌 위상 랜덤
        
        # Object Beam Amplitude (A_O) & Reference Beam Amplitude (A_R)
        # 가우시안 빔 프로파일 등을 흉내내거나, 랜덤 텍스처를 넣을 수 있음
        # 여기서는 간단히 균일하다고 가정하되 약간의 노이즈 추가
        A_O = 0.5 + 0.1 * np.random.rand(H, W) 
        A_R = 0.5 # Reference는 보통 균일
        
        # 결과 저장용 리스트
        imgs_0 = []
        imgs_45 = []
        imgs_90 = []
        imgs_135 = []

        # --- (B) 채널별(RGB) 물리 시뮬레이션 ---
        for ch_idx in range(3): # 0:R, 1:G, 2:B (나중에 merge)
            gamma = gammas[ch_idx]
            
            # 삼각함수 미리 계산
            c = np.cos(gamma / 2)
            s = np.sin(gamma / 2)
            sin_gamma = np.sin(gamma)
            
            # 간섭무늬 물리 모델 (Orthogonal Linear + QWP@45 + Analyzer)
            # Reference(R): Vertical, Object(O): Horizontal 가정 (혹은 그 반대)
            # 일반적인 Polarization Interferometer 식 적용
            
            # I_45, I_135: Gamma 오차에 둔감 (Reference와 Object의 직접 간섭)
            # Phase 0도 (Cos)
            i_45_ch = 0.5 * (A_O**2 + A_R**2 + 2 * A_O * A_R * np.cos(phase_map))
            
            # Phase 180도 (-Cos)
            i_135_ch = 0.5 * (A_O**2 + A_R**2 - 2 * A_O * A_R * np.cos(phase_map))
            
            # I_90, I_0: Gamma 오차에 민감 (Quadrature 성분)
            # Phase 90도 (Sin) -> Sin(Gamma)로 스케일링됨 + DC Offset 발생
            # 이론적 유도 결과 (Input state에 따라 부호는 다를 수 있음)
            i_90_ch = A_O**2 * s**2 + A_R**2 * c**2 + A_O * A_R * sin_gamma * np.sin(phase_map)
            
            # Phase 270도 (-Sin) -> Sin(Gamma)로 스케일링됨 + DC Offset 발생
            i_0_ch  = A_O**2 * c**2 + A_R**2 * s**2 - A_O * A_R * sin_gamma * np.sin(phase_map)

            imgs_0.append(i_0_ch)
            imgs_45.append(i_45_ch)
            imgs_90.append(i_90_ch)
            imgs_135.append(i_135_ch)

        # --- (C) 이미지 합치기 및 저장 ---
        # OpenCV는 BGR 순서로 저장하므로 [B, G, R] 순서로 stack 해야 함
        # 위 루프는 R, G, B 순서였으므로 [imgs[2], imgs[1], imgs[0]]
        
        def merge_and_save(ch_list, angle_name):
            # List [R, G, B] -> Stack -> Transpose to (H, W, 3) -> BGR로 순서 변경
            img_rgb = np.stack(ch_list, axis=-1) # (H, W, 3) RGB
            img_bgr = img_rgb[..., ::-1] # RGB to BGR for OpenCV
            
            # Normalize to 0~255
            img_norm = np.clip(img_bgr * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(f"{save_dir}/{angle_name}_{k}.png", img_norm)
            # cv2.imwrite(f"{save_dir}/{k}_{angle_name}.png", img_norm)

        merge_and_save(imgs_0, "0")
        merge_and_save(imgs_45, "45")
        merge_and_save(imgs_90, "90")
        merge_and_save(imgs_135, "135")

# 사용법
if __name__ == "__main__":
    # LMMSE 코드의 Data/Dataset 경로로 지정
    dataset_path = "Data/Dataset_train" 
    generate_chromatic_hologram_dataset(dataset_path, num_samples=100)