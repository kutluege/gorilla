# 🧠 BFCL — Teknik Mimari ve Değerlendirme Mantığı

## 1. BFCL’nin Temel Felsefesi

BFCL şu soruya cevap arar:

“Model doğru fonksiyonu, doğru parametrelerle, doğru sırada ve doğru sistem durumuna yol açacak şekilde çağırabiliyor mu?”

BFCL:

❌ Prompt formatına bakmaz  
❌ Üsluba bakmaz  
❌ Exact string match istemez  

Ama:

✅ Doğru fonksiyon adı  
✅ Doğru parametre  
✅ Doğru parametre tipi  
✅ Doğru değer  
✅ Doğru state değişimi  
✅ Gereksiz tool çağrısından kaçınma  

konularında deterministik ve katıdır.

---

## 2. Yüksek Seviye Mimari

BFCL iki tamamen ayrık fazdan oluşur:

### A) Generation Phase (bfcl generate)

- JSON test dosyaları okunur.
- Model handler aracılığıyla LLM’e istek atılır.
- Modelin ham çıktısı result/... dosyasına kaydedilir.
- Evaluation yapılmaz.

### B) Evaluation Phase (bfcl evaluate)

- Kaydedilmiş sonuç dosyası okunur.
- Decode edilir (AST veya executable decode).
- Ground truth ile karşılaştırılır.
- Skor hesaplanır.
- LLM ile iletişim yoktur.

---

## 3. Checker Yapısı

BFCL iki ana checker içerir:

### 3.1 AST Checker

- Statik function call doğrulaması yapar.

#### AST Checker Kategorileri

- simple_python
- multiple
- parallel
- parallel_multiple
- live_simple
- live_multiple
- live_parallel
- live_parallel_multiple
- live_irrelevance
- live_relevance
- irrelevance

Bu liste BFCL’nin yalnızca simple/multiple testlerden ibaret olmadığını; live ortam ve relevance testleri de içerdiğini gösterir.

#### AST Checker’a Girenler

- Fonksiyon tanımları (JSON test dosyasından)
- Model çıktısı (handler.decode_ast)
- possible_answer
- Test kategorisi
- Model ismi

### 3.2 Executable Checker

- Simülasyon temelli, stateful değerlendirme yapar.

#### Executable Checker Kategorileri

- multi_turn_base
- multi_turn_miss_func
- multi_turn_miss_param
- multi_turn_long_context
- memory_kv
- memory_vector
- memory_rec_sum
- web_search_*

Bu kategoriler BFCL’nin:

- Multi-turn agent davranışı
- Memory consistency
- Vector store kullanımı
- Recursive summarization
- Web-search agent testleri

gibi ileri seviyeli ajan yeteneklerini de ölçtüğünü gösterir.

---

## 4. AST Checker Detayları

### 4.1 simple

- Tek fonksiyon çağrısı.

**Kontroller:**
- Fonksiyon adı eşleşmesi
- "." → "_" normalizasyonu yapılır
- Required parametre kontrolü
- Tip kontrolü
- Değer kontrolü
- Optional parametre kontrolü

### 4.2 multiple

- Birden fazla tool tanımlı
- Model birini seçmeli
- Fonksiyon adının doğru seçimi kritik

### 4.3 parallel

- Tek tool tanımlı
- Aynı tool birden fazla çağrılmalı
- Sıra önemsiz (order-independent)

### 4.4 parallel_multiple

- Birden fazla farklı tool
- Model birden fazla çağrı yapmalı
- Sıra önemsiz

---

## 5. Tip ve Değer Kontrol Mekanizması

### 5.1 Type Checking

**Python tarafı:**
- int → float kabul edilir
- tuple → list normalize edilir
- Nested list kontrolü 1 seviye derinliğe kadar

**Java / JS tarafı:**
- Model string döndürmeli
- String → type converter ile gerçek tipe çevrilir
- Doğrudan dict/list → type error

### 5.2 String Kontrolü

- Case-insensitive
- Boşluk normalize edilir
- Noktalama toleranslı

### 5.3 List / Dict Kontrolü

**List:** sıra ve değer birebir eşleşmeli

**Dict:**
- Eksik key ❌
- Fazladan key ❌

**List[Dict]:**
- Uzunluk ve sıra birebir eşleşmeli

### 5.4 Variable Heuristiği

Model gerçek değer yerine değişken adı döndürürse:

- Possible answer içinde varsa kabul edilir.
- Tip uyumsuzsa fail.

---

## 6. Generation Pipeline (FC Modu)

### 6.1 Handler Kararı

is_fc_model=True ise:

- inference_single_turn_FC()

çalışır.

### 6.2 Tool Derleme

_compile_tools()  
→ convert_to_tool()

GORILLA şeması:  
→ Gemini / OpenAI tool formatına çevrilir.

### 6.3 Gemini’ye Özgü Konfigürasyon

- AutomaticFunctionCallingConfig(disable=True)
- ThinkingConfig(include_thoughts=True)

Bu şu anlama gelir:

- Native auto function calling kapalı
- Reasoning (thought) ayrı part olarak alınır
- Skorlama reasoning üzerinden değil function_call üzerinden yapılır

### 6.4 Parse

Model yanıtı parçalanır:

- part.function_call
- part.thought
- part.text

Sonuç JSON dosyasına yazılır.

---

## 7. Multi-Turn Execution Mimarisi

BFCL’yi güçlü yapan asıl yapı budur.

### 7.1 involved_classes

Testte hangi backend sınıfların instantiate edileceğini belirtir.

Örn:

- ["TwitterAPI", "GorillaFileSystem"]

### 7.2 initial_config

Sistem başlangıç state’ini tanımlar.

- Dosya sistemi yapısı
- Tweet state’i
- Counter’lar

Test başlarken RAM’de bu state oluşturulur.

### 7.3 Runtime Instance Isolation

Evaluation sırasında:

- Model instance ayrı çalıştırılır.
- Ground truth instance ayrı çalıştırılır.
- State karşılaştırması iki farklı instance arasında yapılır.

Örn:  
- model_instance.root == gt_instance.root ?

Bu isolation deterministik değerlendirme sağlar.

### 7.4 Dynamic Method Binding

Tool çağrıları:

- "find(path='document')"

şeklinde string olarak gelir.

Runtime’da:

- eval("gpt4o_...instance.find(...)")

ile gerçek method çağrılır.

Bu gerçek Python-level execution’dır.

---

## 8. Multi-Turn Checker Mekanizması

Her turn için:

- Model çağrıları execute edilir.
- Ground truth çağrıları execute edilir.
- İki kontrol yapılır:

### 8.1 State Checker

Model ve GT instance state’leri birebir eşleşmeli.  
Tek attribute bile farklıysa fail.

### 8.2 Response Checker (Kümülatif Mantık)

Ground truth çıktıları:  
→ Modelin kümülatif execution sonuçları içinde bulunmalı.

- Model ekstra tool çağırabilir.
- Ama GT çıktısını kaçırırsa fail.
- Superset kabul edilir, subset edilmez.

---

## 9. Multi-Turn Test Varyasyonları

### 9.1 multi_turn_base

Standart multi-turn state testi.

### 9.2 multi_turn_miss_func

Bir turn’de fonksiyon eksik.  
Sonraki turn’de tamamlanıyor.

Ölçtüğü şey:
- API awareness
- Delayed correction

### 9.3 multi_turn_miss_param

Parametre eksik.  
Sonraki turn’de tamamlanıyor.

Ölçtüğü şey:
- Parametre tamamlama adaptasyonu

### 9.4 multi_turn_long_context

Prompt’a:

- Çok uzun
- Alakasız
- Gürültülü API dokümanları

inject edilir.

Test ettiği şey:
- Modelin gürültü içinde doğru fonksiyonu bulabilmesi.

---

## 10. Memory ve Web-Search Testleri

BFCL yalnızca tool çağrısı değil, aynı zamanda:

- memory_kv
- memory_vector
- memory_rec_sum
- web_search_*

testleri içerir.

Bu:

- KV store tutarlılığı
- Vector retrieval
- Recursive summarization
- Web-search tabanlı ajanlık

gibi ileri agent yeteneklerini ölçer.

---

## 11. Skor Hesaplama

accuracy = correct_count / total

Score dosyası:

```json
{
  "accuracy": 0.87,
  "correct_count": 43,
  "total_count": 80
}