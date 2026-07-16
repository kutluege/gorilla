# BFCL Multi-Turn Akışı, Pipeline ve System Prompt Yönetimi (Özet 4)

Bu doküman, BFCL (Berkeley Function Call Leaderboard) altyapısında pipeline'ın test yükleme komutundan (CLI) sonuç dosyası yazımına kadar nasıl çalıştığını, iki ana dosyanın işbirliğini, multi-turn döngülerinin karar noktalarını ve FC (Function Calling) ile Prompting modları arasındaki sistem prompt farklılıklarını açıklar.

## 1. Üst Seviye Pipeline (CLI → Result Dosyası)

Generation sürecinin bir uçtan diğer uca akışı şu adımları izler:
1.  **Başlangıç:** CLI üzerinden girilen `bfcl generate` komutu `bfcl_eval/__main__.py` içindeki `generate()` fonksiyonu ile argümanları toplar ve generation ana akışına delege eder.
2.  **Orkestrasyon:** Akış `bfcl_eval/_llm_response_generation.py` içindeki `main(args)` ile başlar.
3.  **Testlerin Yüklenmesi:** `get_involved_test_entries()` test verilerini kategorisine (veya özel olarak belirtilmiş `run_ids` üzerinden) göre belleğe yükler.
4.  **Eleme ve Belirleme:** `collect_test_cases()`, mevcut üretilmiş sonuçları ve `--allow-overwrite` durumunu dikkate alarak gerçekten üretilecek (eksik veya yeniden yazılacak) test entry'lerinin kesin listesini çıkarır.
5.  **Handler'ın Kurulması:** `build_handler()`, verilen model adına (`model_config.py` tabanlı) uygun model handler sınıfını ayağa kaldırır.
6.  **Paralel Üretim:** `generate_results()`, testleri işlemek için thread havuzu (paralel inference) kurar ve I/O çakışmalarını önlemek için ayrı bir writer thread (yazıcı iş parçacığı) başlatır.
7.  **Inference Sarmalayıcı (Wrapper):** Her bir entry için `multi_threaded_inference()` çağrılır, bu fonksiyon hata yakalamayı üstlenir ve asıl çalışmayı `handler.inference(...)` komutuna yaptırır.
8.  **Sonuçların Yazılması:** Dönen raw yanıtlar ve token/latency gibi metadata verileri, yedeğe alınan kuyruktan alınarak `handler.write(...)` ile modelin kendi altındaki category bazlı result dosyasına eklenir. Nihayetinde tüm işlem bitince JSON dosyaları ID'ye göre sıralı bir şekilde güncellenir.

## 2. İki Ana Dosya Birlikte Nasıl Çalışıyor?

Sistemin kalbi başlıca şu iki dosya etrafında döner:

*   **`bfcl_eval/_llm_response_generation.py` ("İş Akışı Yöneticisi"):** Testleri seçer, bağımlılıkları ayarlar, thread havuzunu kurup yönetir, `base_handler`'ı tetikler ve çıktıyı diske yazar. Generaton sürecinin dış katmanıdır.
*   **`bfcl_eval/model_handler/base_handler.py` ("Inference Motoru"):** Sadece tek bir test entry'si ile ilgilenir. O test case özelinde modele tam olarak ne gönderileceğini (prompt/message format), cevabın nasıl işleneceğini, parse edilip parse edilmediğini ve model içi multi-turn step döngüsünü bizzat yönetir.

## 3. Metot Haritası (Ana Roller)

### `_llm_response_generation.py` Metotları:
*   `main`: Tüm generate sürecinin giriş/çıkış (orchestration) ana noktasıdır.
*   `get_involved_test_entries`: Test verilerinin yüklenmesinden sorumludur.
*   `collect_test_cases`: Testlerin elenip filtrelenmesi (existing result, overwrite, dependency, category filtreleri) görevini üstlenir.
*   `build_handler`: Modelin özelliklerine bakarak -> Handler (örn: `QwenFCHandler`) eşleştirmesini yapar.
*   `generate_results`: Scheduler yapısını çalıştırır ve writer thread'i koordine eder.
*   `multi_threaded_inference`: Tek entry bazlı inference'i sarmalayan, hataların (exceptions) run'ı çökertmesini engelleyen wrapper fonksiyondur.

### `base_handler.py` Metotları:
*   `inference`: En tepe Router'dır. Duruma göre FC/Prompting ve Single/Multi-turn yollarına ayırır.
*   `inference_multi_turn_FC` / `inference_multi_turn_prompting`: Asıl zorlu işin döndüğü turn ve step döngü fonksiyonlarıdır. Tool execution ve state management burada yapılır.
*   `_pre_query_processing_*`, `_compile_tools` (sadece FC), `add_*_message`, `_query_*`, `_parse_query_response_*`: Bunların tamamı model I/O (girdi/çıktı) formatlama zincirini oluşturur.
*   `decode_execute` → `is_empty_execute_response` → `execute_multi_turn_func_call` → `_add_execution_results_*`: Tool-calling döngüsünün çekirdek yapısıdır.
*   `write`: Alınan sonucu kategorisine uygun şekilde formata sokarak yazar.

## 4. Multi-Turn Inference Döngüsü, Veri Gönderimi ve Karar Noktaları

Süreç, `test_entry` üzerinden gelen konuşmaları iki ana döngü içerisinde işler:

*   **Dış Döngü (Turn Loop):** Veri setindeki (`test_entry["question"]`) her bir konuşma turu (turn) için döner.
*   **İç Döngü (Step Loop):** Bir tur içerisinde modelin birden fazla ardışık araç çağırıp sonuçlarını kendi kendine geri beslemesini sağlayan adım döngüsüdür. (`count` değişkeni ile kontrol edilir.)

**Modele Veri Gönderme (State Accumulation) Mantığı:**
*   Handler, tüm sohbet (chat) geçmişini `inference_data["message"]` içinde sürekli biriktirir.
*   `add_first_turn_message_*` ve `_add_next_turn_user_message_*` yeni turn geldiğinde kullanıcının promptunu ekler.
*   `_add_assistant_message_*` modelin step sonundaki cevabını API geçmişine yapıştırır.
*   Eğer bir tool çalışırsa sonucu `_add_execution_results_*` ile `user` veya `tool` rolünde geçmişe eklenip bir sonraki stepte modelin bunu görmesi sağlanır (geri besleme). 

**Kritik Karar / Bitiş Noktaları:**
*   Model modunun FC mi yoksa Prompting mi olacağı `inference()` adlı router ile tayin edilir. 
*   Aynı router, test ID/category bilgilerine bakarak single-turn mu multi-turn mu kararı verir.
*   **Turn Ne Zaman Biter? (Multi-Turn Step Loop'un Kuralları):**
    1.  Modelin ürettiği metinden (decode) tool çıkarılamazsa (yani `is_empty_execute_response=True` dönerse empty decode oluşur), step döngüsü kırılır (`break`) ve bir sonraki tura (turn) geçilir.
    2.  `decode` esnasında exception (veri parse hatası) alınırsa turn yine bitirilir.
    3.  `MAXIMUM_STEP_LIMIT=20` guard'ı aşılırsa kaba kuvvetle "Force quit" çalışır ve işlem başarısız sayılarak doğrudan kırılır.
    *(Not: Turn bitişi, modelin sisteme sihirli bir özel "done token" atmasıyla değil, pratikte "çalıştırılabilir bir tool tag'i/çağrısı bulamadım" sinyali ile anlaşılır.)*

## 5. Gerçek Çıktı Örnekleriyle (Result JSON) Akış İncelemesi

### Örnek 1: `multi_turn_base_0` (Hata + Empty-Response Kapanışı)
*Kaynaklar:* `bfcl_eval/data/BFCL_v4_multi_turn_base.json` ve `result/Qwen_Qwen3-4B-Instruct-2507-FC/multi_turn/...`
Bu örnekte davranış analizi, generation fazının doğru/yanlışı değerlendirmekten ziyade "olanı kaydetmek" üzerine kurulu olduğunu gösterir.
*   **Turn 0 Step 0:** Model `mkdir(temp)` tool komutunu üretir, yürütülür (`execution_result: None`).
*   **Turn 0 Step 1:** Model `mv(final_report.pdf, temp)` üretir. Sistemin sanal ortamında böyle bir dosya bulunmadığı için backend'den "File not found" tarzı gerçek bir hata execution result olarak geri döner.
*   **Turn 0 Step 2:** Model bu hatayı okuduktan sonra yeni bir tool call yapmak yerine, normal düz metin (açıklama) döner. Kodda bu durum boş dizi (`[]`) şeklinde decode edilir. `is_empty_execute_response=True` olduğu için Turn 0 o an sonlandırılır (kapanır).
*   **Turn 1, 2, 3:** Süreç aynı hatalı komut seçimleri ve fail olmuş result'larla ilerler; en sonunda modelin tool vermeyi kestikten sonra ürettiği empty formatla turn'ler teker teker biter.

### Örnek 2: `multi_turn_base_1` (Daha Stabil Tool-Call Zinciri)
Bu örnek, hata alınmadan birbirine bağlanan temiz bir step zincirinin en sonunda nasıl kapandığını gösterir.
*   **Turn 0:** Model önce `pwd`, arkasından `ls(a=True)` gibi çağrıları artarda step bazlı hatasız çalıştırır (`decode -> execute -> add logs -> retry`).
*   İlgili zincirin son step'ine gelindiğinde, model artık istenilen işlemlerin doğrulandığını/bittigini anlayıp final bir yorum metni basar. Yine `decode` işlemi sonucunda çalıştırılabilir hiçbir obje kalmadığı görülür (`[]`) ve o turn başarıyla kapatılıp diğer turlara geçilir. Kapanış mekanizması şaşmaz: **boş decode.**

## 6. FC Modu ve Prompting Modu Arasındaki "System Prompt" Farkları (Qwen Örneği)

Prompting (salt metin işleme) yapabilen model konfigürasyonu ile yerleşik yetenekli FC modunun fonksiyon şemalarını anlama biçimleri büyük farklılık gösterir.

### Prompting Modu (Örn. Qwen Handler base)
Prompting modundaki modellerin native araç çağırma yeteneği olmadığı için her komutu ve JSON veri tiplerini "yazılı bir komut (prompt)" ile anlamaları gerekir.
* `base_oss_handler.py` içinde `_pre_query_processing_prompting` metodu çalıştırılır.
* Bu metod arka planda `utils.py` altındaki `system_prompt_pre_processing_chat_model()` fonksiyonunu tetikler.
* Bu fonksiyon `default_prompts.py` listesinden tam bir "Sen uzman bir foksiyon çağırıcı asistansın, formatı şöyle dönmelisin..." tarzında dayatmalı genel geçer bir sistem prompt metni oluşturur.
* **Sistem Promptu Yoksa:** Test verisinde sistem promptu belirtilmemişse, BFCL'in ürettiği metin listenin başına doğrudan `system` rolü ile eklenir.
* **Sistem Promptu Varsa:** Eğer test verisi default bir sistem prompt içerecek şekilde yüklendiyse, orijinal veriye BFCL'in ürettiği metin `\n\n` bağlacı ile `prepend` edilir ve modele zorla verilir.

### FC Modu (Örn. QwenFCHandler)
FC modunda modeller araçları kendi sistemlerine özel native bir formda, örneğin XML tag'leri arasında (`<tools>...</tools>`) veya argüman json objeleri sayesinde algılar.
* `qwen_fc.py` içerisinde `_pre_query_processing_prompting` özellikle `@override` edilerek, BFCL prompt'unu sağlayan yapı boş döndürülür: `return {"message": [], "function": functions}`.
* Bu ezme (override) işlemi sayesinde BFCL'in yukarıdaki standart system prompt enjeksiyonu **devre dışı bırakılmış (bypassed)** olur.
* **Peki System Prompt hiç mi verilmiyor? (Qwen Özel Durumu):** Qwen'in çalışma prensibi olan ChatML şablonu şöyledir: Qwen, fonksiyon şemalarını API katmanında parametre olarak değil, doğrudan sistem promptunun text'i içine gömülmüş bir `<tools>` XML sarmalı içinde görmek için eğitilmiştir.
* **Sistem Promptu Yoksa:** Qwen'e gereksiz BFCL metinleri yollanmaz ancak, `_format_prompt` sırasında mutlaka bir `<|im_start|>system\n` yaratılır ve içine sadece eldeki `functions` değişkenleri gömülerek (`# Tools\n...<tools>...`) standart sistem prompt iskeleti ayağa kaldırılır.
* **Sistem Promptu Varsa:** Kullanıcının (datasetin) sağladığı asıl sistem promptu, tools satırlarının hemen önüne tertemiz yerleştirilerek modele gönderilir: 
`<|im_start|>system\n{orijinal_dataset_prompt}\n\n# Tools\n...<tools>...`
