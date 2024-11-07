import streamlit as st
import open_clip
import torch
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import chromadb
import logging
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 카테고리 매핑 정의
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

class CustomFashionEmbeddingFunction:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.categories = list(FOLDER_COLLECTION_MAPPING.keys())
    
    def encode_text(self, text):
        """텍스트 인코딩"""
        tokenized = open_clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokenized)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()
    
    def __call__(self, input, categories=None):
        try:
            processed_images = []
            for img in input:
                if isinstance(img, (str, bytes)):
                    img = Image.open(img).convert('RGB')
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img.astype('uint8')).convert('RGB')
                
                processed_img = self.preprocess(img).unsqueeze(0).to(self.device)
                processed_images.append(processed_img)
            
            batch = torch.cat(processed_images)
            
            with torch.no_grad():
                # 이미지 임베딩
                image_features = self.model.encode_image(batch)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 카테고리가 제공된 경우 텍스트 임베딩 결합
                if categories and len(categories) > 0:
                    # 여러 카테고리의 텍스트 임베딩 평균 계산
                    text_features = np.mean([self.encode_text(cat) for cat in categories], axis=0)
                    
                    # 이미지와 텍스트 임베딩 결합
                    combined_features = np.concatenate([
                        image_features.cpu().numpy(),
                        np.repeat(text_features, len(batch), axis=0)
                    ], axis=1)
                    
                    return combined_features
                
                return image_features.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error in embedding function: {e}")
            raise

def search_similar_items(image, mask=None, categories=None, top_k=10):
    """여러 카테고리 정보를 포함한 검색 수행"""
    try:
        client = chromadb.PersistentClient(path="./fashion_multimodal_db")
        embedding_function = CustomFashionEmbeddingFunction()
        
        # 선택된 카테고리에 해당하는 컬렉션만 검색
        collections = []
        if categories and len(categories) > 0:
            selected_collection_names = [FOLDER_COLLECTION_MAPPING[cat] for cat in categories]
            for collection_name in selected_collection_names:
                try:
                    collection = client.get_collection(
                        name=collection_name,
                        embedding_function=embedding_function
                    )
                    collections.append(collection)
                    logger.info(f"Connected to collection: {collection_name}")
                except Exception as e:
                    logger.error(f"Error getting collection {collection_name}: {e}")
                    continue
        else:
            # 카테고리가 선택되지 않은 경우 모든 컬렉션 검색
            collection_names = client.list_collections()
            for collection_info in collection_names:
                try:
                    collection = client.get_collection(
                        name=collection_info.name,
                        embedding_function=embedding_function
                    )
                    collections.append(collection)
                except Exception as e:
                    logger.error(f"Error getting collection {collection_info.name}: {e}")
                    continue
        
        if not collections:
            logger.error("No collections available for search")
            return []
        
        if mask is not None:
            mask_3d = np.stack([mask] * 3, axis=-1)
            masked_image = np.array(image) * mask_3d
            query_image = Image.fromarray(masked_image.astype(np.uint8))
        else:
            query_image = image
        
        all_results = []
        
        for collection in collections:
            try:
                # 이미지와 카테고리 정보를 포함하여 검색
                results = collection.query(
                    query_images=[np.array(query_image)],
                    query_texts=categories if categories else None,
                    n_results=top_k,
                    include=['metadatas', 'distances']
                )
                
                if results and 'metadatas' in results and results['metadatas']:
                    for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                        similarity_score = 1 / (1 + distance)
                        item_data = metadata.copy()
                        item_data['similarity_score'] = similarity_score * 100
                        all_results.append(item_data)
                
            except Exception as e:
                logger.error(f"Error searching in collection: {e}")
                continue
        
        seen_ids = set()
        unique_results = []
        
        for item in sorted(all_results, key=lambda x: x['similarity_score'], reverse=True):
            item_id = item.get('id', '')
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_results.append(item)
                
                if len(unique_results) >= top_k:
                    break
        
        return unique_results
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

def process_search(image, mask, num_results, categories=None):
    """유사 아이템 검색 처리"""
    try:
        with st.spinner("유사한 아이템 검색 중..."):
            similar_items = search_similar_items(image, mask, categories, num_results)
            
        return similar_items
    except Exception as e:
        logger.error(f"Search processing error: {e}")
        return None

def handle_file_upload():
    if st.session_state.uploaded_file is not None:
        image = Image.open(st.session_state.uploaded_file).convert('RGB')
        st.session_state.image = image
        st.session_state.upload_state = 'image_uploaded'
        st.rerun()

def handle_detection():
    if st.session_state.image is not None:
        detected_items = process_segmentation(st.session_state.image)
        st.session_state.detected_items = detected_items
        st.session_state.upload_state = 'items_detected'
        st.rerun()

def handle_search():
    st.session_state.search_clicked = True

def main():
    st.title("패션 이미지 검색")

    # Initialize session state
    if 'selected_categories' not in st.session_state:
        st.session_state.selected_categories = []

    # 카테고리 선택 UI
    st.subheader("검색할 카테고리 선택")
    categories = list(FOLDER_COLLECTION_MAPPING.keys())
    
    # 카테고리를 4열로 표시
    cols = st.columns(4)
    for idx, category in enumerate(categories):
        with cols[idx % 4]:
            if st.checkbox(category, key=f"cat_{category}"):
                if category not in st.session_state.selected_categories:
                    st.session_state.selected_categories.append(category)
            elif category in st.session_state.selected_categories:
                st.session_state.selected_categories.remove(category)

    # 선택된 카테고리 표시
    if st.session_state.selected_categories:
        st.write("선택된 카테고리:", ", ".join(st.session_state.selected_categories))
    
    # 파일 업로더
    if st.session_state.upload_state == 'initial':
        uploaded_file = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'], 
                                       key='uploaded_file', on_change=handle_file_upload)

    # 이미지가 업로드된 상태
    if st.session_state.image is not None:
        st.image(st.session_state.image, caption="업로드된 이미지", use_column_width=True)
        
        if st.session_state.detected_items is None:
            if st.button("의류 검출", key='detect_button', on_click=handle_detection):
                pass
        
        # 검출된 아이템 표시 및 검색
        if st.session_state.detected_items is not None and len(st.session_state.detected_items) > 0:
            cols = st.columns(2)
            for idx, item in enumerate(st.session_state.detected_items):
                with cols[idx % 2]:
                    try:
                        if item.get('mask') is not None:
                            masked_img = np.array(st.session_state.image) * np.expand_dims(item['mask'], axis=2)
                            st.image(masked_img.astype(np.uint8), caption=f"검출된 아이템 {idx + 1}")
                            
                        st.write(f"아이템 {idx + 1}")
                        score = item.get('score')
                        if score is not None and isinstance(score, (int, float)):
                            st.write(f"신뢰도: {score*100:.1f}%")
                        else:
                            st.write("신뢰도: N/A")
                    except Exception as e:
                        logger.error(f"Error displaying item {idx}: {str(e)}")
                        st.error(f"아이템 {idx} 표시 중 오류 발생")
            
            valid_items = [i for i in range(len(st.session_state.detected_items)) 
                          if st.session_state.detected_items[i].get('mask') is not None]
            
            if not valid_items:
                st.warning("검색 가능한 아이템이 없습니다.")
                return
                
            selected_idx = st.selectbox(
                "검색할 아이템 선택:",
                valid_items,
                format_func=lambda i: f"아이템 {i + 1}",
                key='item_selector'
            )
            
            search_col1, search_col2 = st.columns([1, 2])
            with search_col1:
                search_clicked = st.button("유사 아이템 검색", 
                                         key='search_button',
                                         type="primary")
            with search_col2:
                num_results = st.slider("검색 결과 수:", 
                                      min_value=1, 
                                      max_value=20, 
                                      value=5,
                                      key='num_results')

            if search_clicked or st.session_state.get('search_clicked', False):
                st.session_state.search_clicked = True
                selected_item = st.session_state.detected_items[selected_idx]
                
                if selected_item.get('mask') is None:
                    st.error("선택한 아이템에 유효한 마스크가 없습니다.")
                    return
                
                if 'search_results' not in st.session_state:
                    similar_items = process_search(
                        st.session_state.image, 
                        selected_item['mask'], 
                        num_results,
                        categories=st.session_state.selected_categories
                    )
                    st.session_state.search_results = similar_items
                
                if st.session_state.search_results:
                    show_similar_items(st.session_state.search_results)
                else:
                    st.warning("유사한 아이템을 찾지 못했습니다.")

    # 새 검색 버튼
    if st.button("새로운 검색 시작", key='new_search'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()