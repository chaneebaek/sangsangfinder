import os
import re
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VECTOR_DB           = os.getenv("VECTOR_DB", "pinecone").lower()
EMBED_MODEL_PATH    = os.path.join(_BASE_DIR, "models", "embed_finetuned")
SUMMARY_MODEL_PATH  = os.path.join(_BASE_DIR, "models", "summary_finetuned")
CLASSIFY_MODEL_PATH = os.path.join(_BASE_DIR, "models", "classify_finetuned")
BASE_MODEL_EMBED    = os.getenv("BASE_MODEL_EMBED", "jhgan/ko-sroberta-multitask")
EMBEDDER_BACKEND    = os.getenv("EMBEDDER_BACKEND", "sentence-transformers").lower()
SIMCSE_POOLING      = os.getenv("SIMCSE_POOLING", "cls").lower()
SEARCH_ALPHA        = float(os.getenv("SEARCH_ALPHA", "0.5"))
VECTOR_STATE_PATH   = os.getenv(
    "VECTOR_STATE_PATH",
    os.path.join(_BASE_DIR, "pinecone_index"),
)
INDEX_MANIFEST_PATH = os.path.join(VECTOR_STATE_PATH, "index_manifest.json")
NOTICES_CACHE_PATH  = os.path.join(_BASE_DIR, "data", "data_2026.json")

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hansung-notices")
PINECONE_CLOUD      = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION     = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "prod")
PINECONE_CACHE_PATH = os.getenv(
    "PINECONE_CACHE_PATH",
    os.path.join(VECTOR_STATE_PATH, "pinecone_chunks.json"),
)
EMBEDDING_DIM       = int(os.getenv("EMBEDDING_DIM", "768"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY_PAID_TIER = os.getenv("GEMINI_API_KEY_PAID_TIER")
GEMINI_ROUTER_TIMEOUT_SECONDS = int(os.getenv("GEMINI_ROUTER_TIMEOUT_SECONDS", "5"))
GEMINI_REPLY_TIMEOUT_SECONDS = int(os.getenv("GEMINI_REPLY_TIMEOUT_SECONDS", "15"))
GEMINI_FEATURE_TIMEOUT_SECONDS = int(os.getenv("GEMINI_FEATURE_TIMEOUT_SECONDS", "10"))
GEMINI_FEATURE_MODEL = os.getenv("GEMINI_FEATURE_MODEL", "gemini-2.5-flash-lite")
GEMINI_FEATURE_BATCH_SIZE = int(os.getenv("GEMINI_FEATURE_BATCH_SIZE", "8"))
GEMINI_FEATURE_EXTRACTION_ENABLED = os.getenv("GEMINI_FEATURE_EXTRACTION_ENABLED", "true").lower() == "true"
NOTICE_FEATURE_CACHE_PATH = os.getenv(
    "NOTICE_FEATURE_CACHE_PATH",
    os.path.join(_BASE_DIR, "data", "notice_feature_cache.json"),
)

BOARD_LIST_URL = "https://www.hansung.ac.kr/bbs/hansung/2127/artclList.do"
HEADERS        = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TARGET_YEAR    = str(datetime.now().year)

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80

CATEGORIES = [
    "취업/채용", "인턴십", "장학금", "학자금/근로장학", "학사행정",
    "창업", "국제교류", "교육/특강", "비교과", "공모전/경진대회",
    "봉사/서포터즈", "기숙사/생활관", "ROTC", "기타",
]

CATEGORY_PREFIX: dict[str, str] = {
    "채용정보":    "취업/채용",
    "강소기업채용": "취업/채용",
    "인턴쉽":     "인턴십",
    "교외장학금":  "장학금",
    "국가장학금":  "장학금",
    "학자금대출":  "학자금/근로장학",
    "국가근로":    "학자금/근로장학",
    "면학근로":    "학자금/근로장학",
    "공모전":     "공모전/경진대회",
    "정보":       "공모전/경진대회",
}

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "ROTC":           ["ROTC", "학군사관", "학군단", "현역병 모집", "현역병모집", "예비군", "전문사관", "재병역판정검사"],
    "기숙사/생활관":   ["기숙사", "생활관", "상상빌리지", "우촌학사", "임대기숙사", "사감", "입사생 선발", "대학생주택", "학사관"],
    "비교과":         ["비교과", "동아리", "D-School", "포럼", "대동제", "영상제", "입학식",
                      "HS CREW", "상상파크", "라이프 디자인", "문화탐방", "만우절", "오찬 소통",
                      "Lunch with", "천원의 아침밥", "ESG", "진로집단상담", "리더십 탐험",
                      "학생축제", "문화제", "페스티벌", "소모임",
                      "디즈니 프로그램", "디즈니프로그램", "FSU-Disney", "Disney",
                      "새내기 새로배움터", "새로배움터", "총학생회", "사진전", "영상·사진전", "진로 설명회", "트랙 진로"],
    "취업/채용":      ["채용", "신입", "공채", "취업", "채용박람회", "취업박람회", "모집공고", "직무", "채용연계", "일반채용", "추천채용"],
    "인턴십":         ["인턴", "인턴십", "일경험", "체험형", "현장실습", "IPP"],
    "장학금":         ["장학", "장학생", "장학재단", "장학금", "기부장학", "장학사업", "스칼라십", "장학지원"],
    "학자금/근로장학": ["학자금대출", "학자금", "이자지원", "국가근로", "면학근로", "근로장학", "대출이자",
                      "등록금 납부", "등록금 분할", "학기초과자 등록금"],
    "학사행정":       ["수강신청", "수강정정", "졸업", "휴학", "복학", "학점", "트랙변경", "성적", "폐강", "복수전공", "부전공",
                      "재입학", "연계전공", "Micro Degree", "MD과정", "교양영어", "이수신청",
                      "트랙선택", "계절학기", "수업평가", "학위취득유예", "수강포기", "서면신청", "서면 신청",
                      "교차전부", "편입생", "전부(과)", "학위수여식", "학사학위취득",
                      "오리엔테이션", "반편성고사", "합격자 공고", "합격자발표", "선발 결과", "이수 면제",
                      "중간고사", "기말고사", "전공과목 변경", "다전공 신청", "학석사연계",
                      "학사경고", "자기설계전공", "교양필수", "상상력이노베이터"],
    "창업":           ["창업", "창업동아리", "창업지원", "창업멘토링", "스타트업", "아이디어톤", "입주기업", "예비창업", "학생 CEO", "CEO 발굴"],
    "국제교류":       ["교환학생", "어학연수", "파견", "글로벌버디", "국제교류", "해외", "어학", "글로컬",
                      "글로벌 튜터링", "글로벌 Conversation", "글로벌 컨버세이션", "글로벌 문화 소통",
                      "단기연수", "K-Move", "WEST 연수", "한·미대학생", "글로벌 동행"],
    "교육/특강":      ["특강", "교육생", "아카데미", "KDT", "K-디지털", "강좌", "교육과정", "역량강화",
                      "평생교육", "RISE", "마이크로디그리", "TOPCIT", "연구방법론",
                      "초청강연", "특별강연", "핵심역량진단", "HS-CESA", "K-CESA", "UICA", "K-NSSE",
                      "폭력예방교육", "필수교육", "전문과정", "SW마에스트로", "코딩 캠프", "코딩캠프",
                      "연구윤리", "워크숍", "진로 캠프", "심폐소생술", "저작권", "기초역량 가이드"],
    "공모전/경진대회": ["공모전", "경진대회", "챌린지", "해커톤", "대회", "공모", "문학상"],
    "봉사/서포터즈":  ["서포터즈", "서포터스", "봉사", "멘토", "봉사자", "기자단", "자원활동", "멘토단", "멘토링",
                      "홍보대사", "하랑", "소통-e", "앰버서더", "방송국 HBS", "홍보단",
                      "수습기자", "운영자문위원", "모니터링단", "자문단",
                      "바로알림단", "기획단", "체험단", "발굴단", "순찰대", "제작단",
                      "자원지도자", "볼런톤", "청백리포터", "Friends of Korea"],
}

CATEGORY_PATTERN = re.compile(
    r"^(한성공지|국제|학사|비교과|장학|취업|진로|창업|기타|현장실습|교육프로그램|행사|일반공지)\s*"
)
SUFFIX_PATTERN = re.compile(r"\s*(새글|hot|NEW)\s*$", re.IGNORECASE)
