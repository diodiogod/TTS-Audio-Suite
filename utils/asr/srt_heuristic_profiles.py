"""
Language-aware heuristic defaults for subtitle segmentation.
"""

from typing import Dict, Optional


DEFAULT_HEURISTIC_PROFILE_LABEL = "Auto"
CUSTOM_HEURISTIC_PROFILE_LABEL = "Custom"

ENGLISH_DANGLING_TAIL_ALLOWLIST = "a,an,the,to,of,and,or,im,i'm,you,you're,we,they,he,she,it"
ENGLISH_INCOMPLETE_KEYWORDS = "what,why,how,where,who,which,when"

PORTUGUESE_BR_DANGLING_TAIL_ALLOWLIST = (
    "o,a,os,as,um,uma,uns,umas,de,do,da,dos,das,e,ou,mas,se,que,como,quando,"
    "onde,quem,para,pra,por,com,sem,em,no,na,nos,nas,ao,aos,pelo,pela,pelos,pelas"
)
PORTUGUESE_BR_INCOMPLETE_KEYWORDS = (
    "o que,por que,porque,como,onde,quem,qual,quais,quando"
)

_PROFILE_SPECS = [
    {
        "key": "english",
        "label": "English",
        "allowlist": ENGLISH_DANGLING_TAIL_ALLOWLIST,
        "incomplete": ENGLISH_INCOMPLETE_KEYWORDS,
        "aliases": ("en", "en-us", "en-gb", "english"),
    },
    {
        "key": "pt-br",
        "label": "Portuguese (Brazil)",
        "allowlist": PORTUGUESE_BR_DANGLING_TAIL_ALLOWLIST,
        "incomplete": PORTUGUESE_BR_INCOMPLETE_KEYWORDS,
        "aliases": (
            "pt",
            "pt-br",
            "pt_br",
            "portuguese",
            "portuguese (brazil)",
            "brazilian portuguese",
        ),
    },
    {
        "key": "spanish",
        "label": "Spanish",
        "allowlist": (
            "el,la,los,las,un,una,unos,unas,de,del,y,o,pero,si,que,como,cuando,"
            "donde,quien,para,por,con,sin,en,al,a,lo"
        ),
        "incomplete": (
            "lo que,por qué,por que,porque,como,cómo,donde,dónde,quien,quién,cual,cuál,"
            "cuales,cuáles,cuando,cuándo"
        ),
        "aliases": ("es", "es-es", "es-mx", "spanish"),
    },
    {
        "key": "french",
        "label": "French",
        "allowlist": (
            "le,la,les,un,une,des,de,du,et,ou,mais,si,que,comme,quand,qui,pour,"
            "par,avec,sans,en,dans,sur,au,aux,à"
        ),
        "incomplete": "ce que,pourquoi,comment,où,qui,quel,quelle,quels,quelles,quand",
        "aliases": ("fr", "fr-fr", "fr-ca", "french"),
    },
    {
        "key": "italian",
        "label": "Italian",
        "allowlist": (
            "il,lo,la,i,gli,le,un,una,di,del,della,e,o,ma,se,che,come,quando,"
            "dove,chi,per,con,senza,in,su,a,da"
        ),
        "incomplete": "che cosa,perché,perche,come,dove,chi,quale,quali,quando",
        "aliases": ("it", "it-it", "italian"),
    },
    {
        "key": "german",
        "label": "German",
        "allowlist": (
            "der,die,das,ein,eine,und,oder,aber,wenn,dass,wie,wann,wo,wer,für,"
            "mit,ohne,in,an,auf,zu,von,aus,bei,nach"
        ),
        "incomplete": "was,warum,wieso,wie,wo,wer,welche,welcher,welches,wann",
        "aliases": ("de", "de-de", "german"),
    },
    {
        "key": "dutch",
        "label": "Dutch",
        "allowlist": (
            "de,het,een,en,of,maar,als,dat,hoe,wanneer,waar,wie,voor,met,zonder,"
            "in,op,aan,van,bij,naar,uit"
        ),
        "incomplete": "wat,waarom,hoe,waar,wie,welke,wanneer",
        "aliases": ("nl", "nl-nl", "dutch"),
    },
    {
        "key": "russian",
        "label": "Russian",
        "allowlist": (
            "и,а,но,да,или,что,чтобы,как,если,то,в,во,на,за,к,ко,от,до,из,у,о,об,"
            "обо,с,со,по,для,без,при"
        ),
        "incomplete": "что,как,почему,зачем,где,когда,кто,какой,какая,какие,который",
        "aliases": ("ru", "ru-ru", "russian"),
    },
    {
        "key": "romanian",
        "label": "Romanian",
        "allowlist": (
            "și,sau,dar,dacă,că,cum,când,unde,cine,ce,pentru,cu,fără,în,pe,la,de,"
            "din,un,o,niște"
        ),
        "incomplete": "ce,de ce,cum,unde,cine,care,când",
        "aliases": ("ro", "ro-ro", "romanian"),
    },
    {
        "key": "indonesian",
        "label": "Indonesian",
        "allowlist": (
            "dan,atau,tapi,kalau,yang,karena,bahwa,seperti,ketika,di,ke,dari,untuk,"
            "dengan,tanpa,pada,dalam,seorang,sebuah,itu,ini"
        ),
        "incomplete": "apa,kenapa,mengapa,bagaimana,di mana,siapa,yang mana,kapan",
        "aliases": ("id", "id-id", "indonesian", "bahasa indonesia"),
    },
    {
        "key": "malay",
        "label": "Malay",
        "allowlist": (
            "dan,atau,tetapi,kalau,yang,kerana,bahawa,seperti,apabila,di,ke,dari,"
            "untuk,dengan,tanpa,pada,dalam,seorang,sebuah,itu,ini"
        ),
        "incomplete": "apa,kenapa,mengapa,bagaimana,di mana,siapa,yang mana,bila",
        "aliases": ("ms", "ms-my", "malay", "bahasa melayu"),
    },
    {
        "key": "turkish",
        "label": "Turkish",
        "allowlist": "ve,veya,ama,eğer,ki,çünkü,gibi,için,ile,olmadan,da,de,bu,şu,o,bir",
        "incomplete": "ne,neden,niye,nasıl,nerede,kim,hangi,ne zaman",
        "aliases": ("tr", "tr-tr", "turkish"),
    },
    {
        "key": "polish",
        "label": "Polish",
        "allowlist": (
            "i,lub,ale,że,żeby,jak,kiedy,gdzie,kto,co,dla,z,bez,w,na,do,od,po,"
            "przy,o,u"
        ),
        "incomplete": "co,dlaczego,jak,gdzie,kto,który,która,które,kiedy",
        "aliases": ("pl", "pl-pl", "polish"),
    },
    {
        "key": "czech",
        "label": "Czech",
        "allowlist": "a,nebo,ale,že,aby,jak,když,kde,kdo,co,pro,s,bez,v,na,do,od,po,u,o",
        "incomplete": "co,proč,jak,kde,kdo,který,která,které,kdy",
        "aliases": ("cs", "cs-cz", "czech"),
    },
    {
        "key": "swedish",
        "label": "Swedish",
        "allowlist": (
            "och,eller,men,om,att,som,när,var,vem,för,med,utan,i,på,till,från,av,"
            "en,ett,den,det"
        ),
        "incomplete": "vad,varför,hur,var,vem,vilken,vilka,när",
        "aliases": ("sv", "sv-se", "swedish"),
    },
    {
        "key": "danish",
        "label": "Danish",
        "allowlist": (
            "og,eller,men,hvis,at,som,når,hvor,hvem,for,med,uden,i,på,til,fra,af,"
            "en,et,den,det"
        ),
        "incomplete": "hvad,hvorfor,hvordan,hvor,hvem,hvilken,hvilke,hvornår",
        "aliases": ("da", "da-dk", "danish"),
    },
    {
        "key": "finnish",
        "label": "Finnish",
        "allowlist": "ja,tai,mutta,jos,että,kuten,kun,missä,kuka,mikä,sekä,eli",
        "incomplete": "mikä,miksi,miten,missä,kuka,kumpi,milloin",
        "aliases": ("fi", "fi-fi", "finnish"),
    },
    {
        "key": "greek",
        "label": "Greek",
        "allowlist": (
            "και,ή,αλλά,αν,ότι,πως,όπως,όταν,όπου,ποιος,για,με,χωρίς,σε,από,προς,"
            "στο,στη,στον,στην,το,τη,τον,την,ένα,μια"
        ),
        "incomplete": "τι,γιατί,πώς,πού,ποιος,ποια,ποιο,ποιοι,πότε",
        "aliases": ("el", "el-gr", "greek"),
    },
]

HEURISTIC_PROFILE_OPTIONS = [
    DEFAULT_HEURISTIC_PROFILE_LABEL,
    *[spec["label"] for spec in _PROFILE_SPECS],
    CUSTOM_HEURISTIC_PROFILE_LABEL,
]

_PROFILE_DEFAULTS: Dict[str, Dict[str, str]] = {
    spec["key"]: {
        "merge_dangling_tail_allowlist": spec["allowlist"],
        "merge_incomplete_keywords": spec["incomplete"],
    }
    for spec in _PROFILE_SPECS
}

_PROFILE_LABEL_TO_KEY = {
    DEFAULT_HEURISTIC_PROFILE_LABEL.lower(): "auto",
    CUSTOM_HEURISTIC_PROFILE_LABEL.lower(): "custom",
    **{spec["label"].lower(): spec["key"] for spec in _PROFILE_SPECS},
}

_LANGUAGE_TO_PROFILE_KEY = {
    alias: spec["key"]
    for spec in _PROFILE_SPECS
    for alias in spec["aliases"]
}

_LANGUAGE_PREFIXES = (
    ("pt", "pt-br"),
    ("en", "english"),
    ("es", "spanish"),
    ("fr", "french"),
    ("it", "italian"),
    ("de", "german"),
    ("nl", "dutch"),
    ("ru", "russian"),
    ("ro", "romanian"),
    ("id", "indonesian"),
    ("ms", "malay"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("cs", "czech"),
    ("sv", "swedish"),
    ("da", "danish"),
    ("fi", "finnish"),
    ("el", "greek"),
)


def normalize_profile_selection(selection: Optional[str]) -> str:
    if not selection:
        return "auto"
    return _PROFILE_LABEL_TO_KEY.get(str(selection).strip().lower(), "auto")


def resolve_profile_from_language(language: Optional[str]) -> str:
    if not language:
        return "english"

    normalized = str(language).strip().lower().replace("_", "-")
    if normalized in _LANGUAGE_TO_PROFILE_KEY:
        return _LANGUAGE_TO_PROFILE_KEY[normalized]

    for prefix, profile_key in _LANGUAGE_PREFIXES:
        if normalized.startswith(f"{prefix}-"):
            return profile_key

    return "english"


def get_profile_defaults(profile_key: str) -> Dict[str, str]:
    return dict(_PROFILE_DEFAULTS.get(profile_key, _PROFILE_DEFAULTS["english"]))


def resolve_profile_defaults(selection: Optional[str], language: Optional[str] = None) -> Dict[str, str]:
    normalized = normalize_profile_selection(selection)
    if normalized == "custom":
        return {}
    if normalized == "auto":
        normalized = resolve_profile_from_language(language)
    return get_profile_defaults(normalized)
