import streamlit as st
import open_clip
import torch
from PIL import Image
import numpy as np
import chromadb
import logging
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CLIP 임베딩 함수
class CustomFashionEmbeddingFunction:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def __call__(self, input):
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
                features = self.model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.cpu().numpy()
            
            return features
            
        except Exception as e:
            logger.error(f"Error in embedding function: {e}")
            raise

def search_similar_items(image, top_k=10):
    """여러 컬렉션에서 검색 수행"""
    try:
        # ChromaDB 클라이언트 설정
        client = chromadb.PersistentClient(path="./fashion_multimodal_db")
        embedding_function = CustomFashionEmbeddingFunction()
        
        # 모든 컬렉션 가져오기
        collections = []
        collection_names = client.list_collections()
        
        for collection_info in collection_names:
            try:
                collection = client.get_collection(
                    name=collection_info.name,
                    embedding_function=embedding_function
                )
                collections.append(collection)
                logger.info(f"Connected to collection: {collection_info.name}")
            except Exception as e:
                logger.error(f"Error getting collection {collection_info.name}: {e}")
                continue
        
        if not collections:
            logger.error("No collections available for search")
            return []
        
        # 각 컬렉션에서 검색 수행
        all_results = []
        
        for collection in collections:
            try:
                results = collection.query(
                    query_images=[np.array(image)],
                    n_results=top_k,
                    include=['metadatas', 'distances']
                )
                
                if results and 'metadatas' in results and results['metadatas']:
                    for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                        similarity_score = 1 / (1 + distance)  # 거리를 유사도 점수로 변환
                        
                        item_data = metadata.copy()
                        item_data['similarity_score'] = similarity_score * 100  # 백분율로 변환
                        all_results.append(item_data)
                
            except Exception as e:
                logger.error(f"Error searching in collection: {e}")
                continue
        
        # 결과 정렬 및 중복 제거
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

def show_similar_items(similar_items):
    """유사 아이템 표시"""
    if not similar_items:
        st.warning("유사한 아이템을 찾지 못했습니다.")
        return
        
    st.subheader("유사한 아이템:")
    
    items_per_row = 2
    for i in range(0, len(similar_items), items_per_row):
        cols = st.columns(items_per_row)
        for j, col in enumerate(cols):
            if i + j < len(similar_items):
                item = similar_items[i + j]
                with col:
                    try:
                        if 'uri' in item:
                            st.image(item['uri'], use_column_width=True)
                        
                        st.markdown(f"**유사도: {item['similarity_score']:.1f}%**")
                        
                        st.write(f"카테고리: {item.get('category', '알 수 없음')}")
                        if 'collection' in item:
                            st.write(f"컬렉션: {item['collection']}")
                        
                        name = item.get('name', '알 수 없음')
                        if len(name) > 50:
                            name = name[:47] + "..."
                        st.write(f"이름: {name}")
                        
                        st.divider()
                        
                    except Exception as e:
                        logger.error(f"Error displaying item: {e}")
                        st.error("이 아이템을 표시하는 중 오류가 발생했습니다")

def process_search(image, num_results):
    """유사 아이템 검색 처리"""
    try:
        with st.spinner("유사한 아이템 검색 중..."):
            similar_items = search_similar_items(image, num_results)
            
        return similar_items
    except Exception as e:
        logger.error(f"Search processing error: {e}")
        return None

def main():
    st.title("패션 이미지 검색")

    # 파일 업로더
    uploaded_file = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # 이미지 표시
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="업로드된 이미지", use_column_width=True)
        
        # 검색 옵션
        col1, col2 = st.columns([1, 2])
        with col1:
            search_button = st.button("유사 아이템 검색", type="primary")
        with col2:
            num_results = st.slider("검색 결과 수:", 
                                  min_value=1, 
                                  max_value=20, 
                                  value=5)

        # 검색 실행
        if search_button:
            similar_items = process_search(image, num_results)
            if similar_items:
                show_similar_items(similar_items)
            else:
                st.warning("유사한 아이템을 찾지 못했습니다.")

    # 새 검색 버튼
    if st.button("새로운 검색 시작"):
        st.rerun()

if __name__ == "__main__":
    main()