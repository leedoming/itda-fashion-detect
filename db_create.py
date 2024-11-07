import streamlit as st
import chromadb
import logging
import open_clip
import torch
from PIL import Image
import os
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import shutil
import numpy as np

# 폴더명과 컬렉션명 매핑
FOLDER_COLLECTION_MAPPING = {
    "가디건": "kardigun",
    "긴팔": "longsleeve",
    "나일론": "pants_nylon",
    "니트": "knit",
    "레깅스": "pants_leggings",
    "맨투맨": "sweatshirt",
    "바지": "pants_etc",
    "반팔": "shortsleeve",
    "벌룬핏": "pants_balloonfit",
    "베스트, 나시": "vest_tanktop",
    "부츠컷": "pants_bootscut",
    "셔츠": "shirts",
    "쇼츠": "pants_shorts",
    "스웨트팬츠": "pants_sweat",
    "스커트팬츠": "pants_skirt",
    "슬랙스": "pants_slacks",
    "슬림핏": "pants_slimfit",
    "와이드": "pants_wide",
    "운동복": "training",
    "원피스": "dress",
    "자켓": "jacket",
    "점프수트": "jumpsuit",
    "조거팬츠": "pants_jogger",
    "청바지": "pants_jean",
    "치마": "skirt",
    "코듀로이": "pants_corduroy",
    "코트": "coat",
    "파라슈트, 카고": "pants_parachute_cargo",
    "후드티": "hoodie"
}

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fashion_db_addition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def reset_database():
    """데이터베이스 초기화"""
    try:
        shutil.rmtree("./fashion_multimodal_db", ignore_errors=True)
        logger.info("데이터베이스가 초기화되었습니다.")
    except Exception as e:
        logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")

def load_models():
    """모델 로드 함수"""
    try:
        logger.info("모델 로딩 중...")
        model, preprocess_val, _ = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, preprocess_val, device
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {str(e)}")
        raise

def create_metadata(file_path, collection_name):
    """메타데이터 생성 함수"""
    try:
        folder_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 고유 ID 생성 (파일명 해시)
        unique_id = str(abs(hash(file_name)))[:10]
        
        return {
            'id': unique_id,
            'category': folder_name,
            'collection': collection_name,
            'name': file_name,
            'uri': file_path
        }
    except Exception as e:
        logger.error(f"메타데이터 생성 중 오류 발생: {str(e)}")
        return None

def load_image_from_file(image_path):
    """이미지 파일 로드 함수"""
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception as e:
        logger.error(f"이미지 로드 실패 ({image_path}): {str(e)}")
        return None

def get_folder_image_files(root_dir):
    """폴더별 이미지 파일 리스트 생성 함수"""
    folder_images = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
    
    for folder_name in FOLDER_COLLECTION_MAPPING.keys():
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.exists(folder_path):
            folder_images[folder_name] = []
            for file in os.listdir(folder_path):
                if os.path.splitext(file)[1].lower() in image_extensions:
                    folder_images[folder_name].append(os.path.join(folder_path, file))
    
    return folder_images

def process_batch(image_paths, collection_name, model, preprocess_val, device, progress_placeholder=None, batch_size=32):
    """배치 단위로 이미지 처리"""
    processed_items = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_metadata = []
        valid_indices = []

        for idx, path in enumerate(batch_paths):
            metadata = create_metadata(path, collection_name)
            if metadata is None:
                continue
                
            image = load_image_from_file(path)
            if image is None:
                continue
                
            try:
                image_tensor = preprocess_val(image).unsqueeze(0)
                batch_images.append(image_tensor)
                batch_metadata.append(metadata)
                valid_indices.append(idx)
                
                if progress_placeholder:
                    with progress_placeholder.container():
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(image, caption=metadata['name'], use_column_width=True)
                        with col2:
                            st.write(f"컬렉션: {metadata['collection']}")
                            st.write(f"카테고리: {metadata['category']}")
                            st.write(f"제품명: {metadata['name']}")
                            st.write(f"ID: {metadata['id']}")
                            
            except Exception as e:
                logger.error(f"이미지 처리 중 오류 발생 ({path}): {str(e)}")
                continue
        
        if not batch_images:
            continue
            
        try:
            batch_tensor = torch.cat(batch_images).to(device)
            
            with torch.no_grad():
                features = model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.cpu().numpy()
            
            for idx, feature in enumerate(features):
                processed_items.append({
                    'id': batch_metadata[idx]['id'],
                    'embedding': feature.tolist(),
                    'metadata': batch_metadata[idx],
                    'uri': batch_metadata[idx]['uri']
                })
                
        except Exception as e:
            logger.error(f"배치 처리 중 오류 발생: {str(e)}")
            continue
            
    return processed_items

def add_to_collection(collection, processed_items):
    """배치 단위로 컬렉션에 추가"""
    if not processed_items:
        return 0
        
    try:
        collection.add(
            ids=[item['id'] for item in processed_items],
            embeddings=[item['embedding'] for item in processed_items],
            metadatas=[item['metadata'] for item in processed_items],
            uris=[item['uri'] for item in processed_items]
        )
        return len(processed_items)
            
    except Exception as e:
        logger.error(f"컬렉션 추가 중 오류 발생: {str(e)}")
        return 0

def main():
    st.set_page_config(layout="wide")
    st.title("패션 이미지 임베딩 프로세스")
    
    # 이미지 루트 폴더 경로 입력
    root_dir = st.text_input("이미지 루트 폴더 경로를 입력하세요")
    
    # DB 초기화 옵션
    if st.checkbox("데이터베이스 초기화"):
        if st.button("데이터베이스 초기화 실행"):
            reset_database()
            st.success("데이터베이스가 초기화되었습니다.")
    
    if not root_dir or not os.path.exists(root_dir):
        st.warning("유효한 폴더 경로를 입력해주세요.")
        return
    
    if st.button("임베딩 시작"):
        try:
            # 진행 상황 표시 영역
            overall_progress = st.progress(0)
            status_text = st.empty()
            progress_placeholder = st.empty()
            
            # 모델 로드
            status_text.text("모델 로딩 중...")
            model, preprocess_val, device = load_models()
            
            # ChromaDB 설정
            client = chromadb.PersistentClient(path="./fashion_multimodal_db")
            embedding_function = OpenCLIPEmbeddingFunction()
            data_loader = ImageLoader()
            
            # 폴더별 이미지 파일 가져오기
            folder_images = get_folder_image_files(root_dir)
            total_folders = len(folder_images)
            
            if total_folders == 0:
                st.warning("처리할 이미지 폴더가 없습니다.")
                return
            
            # 전체 통계 초기화
            total_stats = {'total': 0, 'processed': 0, 'added': 0, 'failed': 0}
            
            # 폴더별 처리
            for folder_idx, (folder_name, image_files) in enumerate(folder_images.items()):
                collection_name = FOLDER_COLLECTION_MAPPING[folder_name]
                
                # 컬렉션 생성/로드
                collection = client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=embedding_function,
                    data_loader=data_loader
                )
                
                total_files = len(image_files)
                total_stats['total'] += total_files
                
                if total_files == 0:
                    continue
                
                status_text.text(f"처리중... 폴더: {folder_name} ({folder_idx + 1}/{total_folders})")
                
                # 배치 크기 설정
                batch_size = 32
                
                # 배치 단위로 처리
                for i in range(0, total_files, batch_size):
                    batch_paths = image_files[i:i + batch_size]
                    processed_items = process_batch(
                        batch_paths,
                        collection_name,
                        model,
                        preprocess_val,
                        device,
                        progress_placeholder,
                        batch_size
                    )
                    
                    added_count = add_to_collection(collection, processed_items)
                    
                    total_stats['processed'] += len(batch_paths)
                    total_stats['added'] += added_count
                    total_stats['failed'] += len(batch_paths) - added_count
                    
                    # 전체 진행률 업데이트
                    overall_progress.progress((total_stats['processed']) / total_stats['total'])
                    status_text.text(
                        f"처리중... 폴더: {folder_name} ({folder_idx + 1}/{total_folders}) "
                        f"파일: {i + len(batch_paths)}/{total_files} "
                        f"(전체 성공: {total_stats['added']}, "
                        f"실패: {total_stats['failed']})"
                    )
            
            # 최종 결과 표시
            st.success("모든 폴더 처리 완료!")
            st.write("### 처리 결과")
            st.write(f"- 전체 파일: {total_stats['total']}")
            st.write(f"- 처리된 파일: {total_stats['processed']}")
            st.write(f"- 추가된 파일: {total_stats['added']}")
            st.write(f"- 실패한 파일: {total_stats['failed']}")
            
        except Exception as e:
            st.error(f"처리 중 오류 발생: {str(e)}")
            logger.error(f"처리 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()