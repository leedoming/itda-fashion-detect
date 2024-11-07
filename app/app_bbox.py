import streamlit as st
import open_clip
import torch
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
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

# Initialize session state
if 'image' not in st.session_state:
    st.session_state.image = None
if 'detected_boxes' not in st.session_state:
    st.session_state.detected_boxes = None
if 'selected_box_index' not in st.session_state:
    st.session_state.selected_box_index = None
if 'cropped_image' not in st.session_state:
    st.session_state.cropped_image = None

# Load segmentation model
@st.cache_resource
def load_segmentation_model():
    try:
        model_name = "mattmdjaga/segformer_b2_clothes"
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.to('cuda')
            
        return model, image_processor
    except Exception as e:
        logger.error(f"Error loading segmentation model: {e}")
        raise

def get_bounding_boxes(mask):
    """마스크에서 바운딩 박스 추출"""
    try:
        # 레이블링된 컴포넌트 찾기
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(mask)
        
        boxes = []
        for i in range(1, num_features + 1):
            # 각 레이블에 대한 바운딩 박스 좌표 찾기
            y_indices, x_indices = np.where(labeled_array == i)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            # 마진 추가 (10%)
            margin = 0.1
            height = y_indices.max() - y_indices.min()
            width = x_indices.max() - x_indices.min()
            
            y_min = max(0, int(y_indices.min() - height * margin))
            y_max = min(mask.shape[0], int(y_indices.max() + height * margin))
            x_min = max(0, int(x_indices.min() - width * margin))
            x_max = min(mask.shape[1], int(x_indices.max() + width * margin))
            
            boxes.append({
                'bbox': (x_min, y_min, x_max, y_max),
                'area': (y_max - y_min) * (x_max - x_min)
            })
        
        # 면적 기준 내림차순 정렬
        boxes.sort(key=lambda x: x['area'], reverse=True)
        return boxes
        
    except Exception as e:
        logger.error(f"Error getting bounding boxes: {e}")
        return []

def detect_fashion_items(image):
    """의류 아이템 감지 및 바운딩 박스 추출"""
    try:
        model, image_processor = load_segmentation_model()
        
        # 이미지 전처리
        inputs = image_processor(image, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # 추론
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 크기 복원
        logits = outputs.logits.cpu()
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        # 세그멘테이션 마스크 생성
        seg_masks = upsampled_logits.argmax(dim=1).numpy()
        
        # 배경이 아닌 모든 클래스에 대한 통합 마스크
        clothes_mask = (seg_masks[0] > 0).astype(float)
        
        # 바운딩 박스 추출
        boxes = get_bounding_boxes(clothes_mask)
        
        return boxes, clothes_mask
        
    except Exception as e:
        logger.error(f"Error in fashion detection: {e}")
        return [], None

def search_similar_items(image, top_k=10):
    """여러 컬렉션에서 검색 수행"""
    try:
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
                        similarity_score = 1 / (1 + distance)
                        
                        item_data = metadata.copy()
                        item_data['similarity_score'] = similarity_score * 100
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

def crop_image(image, bbox):
    """이미지를 바운딩 박스 기준으로 크롭"""
    x_min, y_min, x_max, y_max = bbox
    return image.crop((x_min, y_min, x_max, y_max))

def main():
    st.title("패션 이미지 검색")

    # 파일 업로더
    uploaded_file = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # 이미지 로드 및 표시
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="업로드된 이미지", use_column_width=True)
        
        # 의류 감지
        with st.spinner("의류 영역 감지 중..."):
            boxes, mask = detect_fashion_items(image)
        
        if not boxes:
            st.warning("의류 아이템을 찾지 못했습니다.")
            return
        
        # 감지된 영역 표시 및 선택
        st.subheader("감지된 의류 아이템")
        
        cols = st.columns(min(len(boxes), 3))
        for idx, (box, col) in enumerate(zip(boxes, cols)):
            bbox = box['bbox']
            cropped = crop_image(image, bbox)
            with col:
                st.image(cropped, caption=f"아이템 {idx + 1}", use_column_width=True)
        
        # 아이템 선택
        selected_idx = st.selectbox(
            "검색할 아이템 선택:",
            range(len(boxes)),
            format_func=lambda x: f"아이템 {x + 1}"
        )
        
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
            selected_box = boxes[selected_idx]['bbox']
            cropped_image = crop_image(image, selected_box)
            
            with st.spinner("유사한 아이템 검색 중..."):
                similar_items = search_similar_items(cropped_image, num_results)
                
            if similar_items:
                show_similar_items(similar_items)
            else:
                st.warning("유사한 아이템을 찾지 못했습니다.")

    # 새 검색 버튼
    if st.button("새로운 검색 시작"):
        st.rerun()

if __name__ == "__main__":
    main()
