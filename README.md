# Optimasi Rute Distribusi Obat ke Puskesmas di Kota Surabaya

**Studi kasus Machine Learning II: pengelompokan puskesmas dan optimasi urutan kunjungan menggunakan jaringan jalan, Data Augmentation, dan Q-Learning.**

Proyek ini menyusun rute distribusi obat dari **Dinas Kesehatan Kota Surabaya** menuju **63 puskesmas**. Seluruh lokasi dibagi menjadi beberapa kelompok distribusi, kemudian urutan kunjungan pada setiap kelompok dioptimasi berdasarkan estimasi waktu perjalanan pada jaringan jalan.

> **Status:** proyek penelitian/eksperimen akademik. Rute dan waktu perjalanan merupakan hasil perhitungan serta simulasi, **bukan** jadwal distribusi resmi atau hasil uji kendaraan di lapangan.

## Tim dan pembagian tugas

| Anggota | Tanggung jawab |
|---|---|
| **Vani** | Pengumpulan data lokasi (*fetch data*), pengolahan jaringan jalan dan Dijkstra, serta **Data Augmentation** |
| **Anggota 2** | **Clustering** (Agglomerative vs K-Medoids), pengembangan **Q-Learning**, tuning, dan evaluasi terhadap Nearest Neighbor |

> Ganti `Anggota 2` dengan nama anggota yang mengerjakan clustering dan RL sebelum memublikasikan repository.

## Tujuan penelitian

- Membentuk kelompok distribusi berdasarkan **jarak jaringan jalan**, bukan jarak lurus geografis.
- Membandingkan **Agglomerative Clustering** dan **K-Medoids** untuk 11 kelompok distribusi.
- Membuat skenario perubahan waktu perjalanan melalui **Data Augmentation**.
- Mengoptimalkan urutan kunjungan dengan **tabular Q-Learning** dan membandingkannya dengan **Nearest Neighbor (NN)**.
- Mengamati pengaruh jumlah episode, variasi data training, dan pemilihan *checkpoint* terhadap kualitas rute.

## Pipeline penelitian

```text
Google Places API / koordinat lokasi
               |
               v
    63 Puskesmas + 1 Dinkes
               |
               v
  OpenStreetMap + OSMnx + NetworkX
               |
               v
     Graf jalan berarah
               |
               v
      Dijkstra shortest path
        /               \
       v                 v
Matriks jarak       Matriks waktu
     64 x 64           64 x 64
       |                 |
       v                 v
Clustering          Data Augmentation
Agglomerative       20 skenario S00-S19
vs K-Medoids             |
       |                 |
       +--------+--------+
                v
       11 kelompok distribusi
                |
                v
     Q-Learning dan Nearest Neighbor
                |
                v
    Evaluasi total waktu perjalanan
                |
                v
 Dinkes -> anggota cluster -> Dinkes
```

**Catatan:** pengelompokan menggunakan matriks **jarak**, sedangkan optimasi rute dan evaluasinya menggunakan matriks **waktu perjalanan**. Dinkes merupakan depot, tidak ikut menjadi anggota cluster.

## Metodologi

### 1. Pengumpulan data dan matriks Dijkstra

- Mengumpulkan nama serta koordinat lokasi puskesmas dan Dinkes melalui Google Places API.
- Mengambil jaringan jalan yang dapat dilalui kendaraan dari OpenStreetMap menggunakan **OSMnx**.
- Mempertahankan informasi arah jalan pada graf berarah dan mengambil komponen jaringan utama yang terhubung.
- Menghubungkan setiap lokasi ke node jalan terdekat (*node snapping*).
- Menghitung matriks **jarak (meter)** dan **waktu (detik)** antarlokasi menggunakan **Dijkstra**.

Hasil utamanya adalah `lokasi_osm_nodes.csv`, `matrix_length.npy`, dan `matrix_time.npy`, dengan matriks berukuran **64 × 64** (63 puskesmas + 1 Dinkes).

### 2. Clustering: Agglomerative vs K-Medoids

Sebanyak 63 puskesmas dikelompokkan menggunakan matriks jarak hasil Dijkstra. Karena graf jalan berarah, jarak perjalanan pergi dan pulang dirata-ratakan untuk membuat matriks jarak simetris.

Parameter yang digunakan:

| Parameter | Pengaturan |
|---|---|
| Jumlah kelompok | **k = 11** (ditetapkan untuk kedua metode) |
| Batas anggota | Maksimal **12 puskesmas per kelompok** |
| Agglomerative | `average` dan `complete` linkage |
| K-Medoids | 10 random seed (42–51), pembaruan medoid bergantian |
| Evaluasi utama | **Silhouette Score** dengan jarak `precomputed` |
| Evaluasi pendukung | **Calinski-Harabasz Index** pada fitur hasil **MDS 5 dimensi** |

Konfigurasi hanya dipertimbangkan jika menghasilkan 11 cluster dan memenuhi batas anggota. Silhouette diprioritaskan dalam pemilihan konfigurasi; Calinski-Harabasz digunakan sebagai metrik pendukung.

**Hasil perbandingan:**

| Metrik | Agglomerative (*complete*) | K-Medoids (seed 42) |
|---|---:|---:|
| Silhouette Score ↑ | **0,3350** | 0,2899 |
| Calinski-Harabasz ↑ | **46,2743** | 36,1100 |
| Jumlah cluster | 11 | 11 |
| Anggota cluster terbesar | 9 | 10 |
| Puskesmas terkelompok | 63/63 | 63/63 |

**Metode yang dipilih adalah Agglomerative Clustering dengan *complete linkage*.** Nilai `k = 11` merupakan parameter eksperimen yang ditetapkan, **bukan** klaim bahwa 11 adalah jumlah cluster optimal berdasarkan Elbow Method.

### 3. Data Augmentation waktu perjalanan

Data Augmentation menyimulasikan perubahan waktu perjalanan dengan memberikan tingkat perlambatan pada kategori jalan **lokal, kolektor, arteri, dan tol**.

- Memeriksa dan memperbaiki atribut `travel_time` yang tidak valid pada graf.
- Membuat **20 skenario** perjalanan, dari **S00 (normal)** sampai **S19**.
- Menghitung kembali matriks waktu perjalanan untuk setiap skenario.
- Menyimpan hasil sebagai `data_augmentation/matrix_time_aug.npy` dengan ukuran **(20, 64, 64)**.

Skenario merupakan **simulasi**, bukan rekaman lalu lintas waktu nyata.

### 4. Q-Learning dan baseline Nearest Neighbor

Setiap cluster membentuk masalah rute tersendiri dengan depot yang sama, yakni Dinkes.

| Komponen RL | Definisi |
|---|---|
| Agent | Kendaraan distribusi |
| State | `(posisi, visited_bitmask)` |
| Action | Puskesmas yang belum dikunjungi; kembali ke depot setelah seluruhnya dikunjungi |
| Reward | Negatif waktu perjalanan dalam menit |
| Algoritma | **Tabular Q-Learning** dengan ε-greedy |
| Baseline | **Nearest Neighbor**, memilih lokasi berikutnya dengan waktu tempuh terkecil |

Pada tuning tahap kedua, konfigurasi yang dibandingkan adalah:

| Parameter | Nilai |
|---|---|
| Episode | **12.000** dan **24.000** |
| Learning rate (α) | **0,10** |
| Discount factor (γ) | **1,00** |
| Random seed | **42** |
| Skenario training | **S00** atau **S00–S11** |
| Validasi | **S12–S15** |
| Testing | **S16–S19** |
| Checkpoint | Evaluasi rute greedy setiap **4.000 episode** |

Q-Learning menggunakan penurunan epsilon secara linear dan menyimpan rute *checkpoint* dengan total waktu terendah menurut **data training**. Baseline NN dibangun dari data training yang sama, sehingga perbandingan setara dari sisi informasi skenario yang tersedia. Perbandingan ini **tidak menyamakan biaya komputasi** NN dan RL.

## Hasil eksperimen

### Hasil validasi tuning

Nilai merupakan **rata-rata jumlah durasi perjalanan seluruh 11 kelompok**, dalam menit. **Semakin kecil semakin baik.**

| Data training | Episode | RL Checkpoint | NN setara | Penghematan RL vs NN |
|---|---:|---:|---:|---:|
| S00 | 12.000 | 612,81 | 602,30 | -1,74% |
| S00–S11 | 12.000 | 618,85 | 602,30 | -2,75% |
| **S00** | **24.000** | **585,26** | **602,30** | **+2,83%** |
| S00–S11 | 24.000 | 593,36 | 602,30 | +1,48% |

Konfigurasi dengan nilai validasi terendah adalah **24.000 episode, α = 0,10, training S00, menggunakan rute checkpoint terbaik**.

### Hasil testing pada S16–S19

| Skenario | Nearest Neighbor | Q-Learning Checkpoint | Penghematan RL |
|---|---:|---:|---:|
| S16 | 538,26 | **521,39** | 3,13% |
| S17 | 631,72 | **615,11** | 2,63% |
| S18 | 601,13 | **587,57** | 2,26% |
| S19 | 704,35 | **684,65** | 2,80% |
| **Rata-rata** | **618,87** | **602,18** | **2,70%** |

Pada empat skenario testing simulasi, **Q-Learning Checkpoint menghasilkan rata-rata total waktu 16,69 menit (2,70%) lebih rendah daripada Nearest Neighbor**.

Sebagai pembanding, kebijakan Q-Learning yang langsung diambil dari **Q-table pada akhir training** menghasilkan rata-rata **663,56 menit**. Perbedaan ini menunjukkan pentingnya strategi pemilihan checkpoint pada eksperimen tersebut, walaupun pengaruh checkpoint belum diisolasi dari perubahan strategi training lainnya.

> Seluruh angka waktu di atas merupakan **penjumlahan durasi 11 rute kelompok**, bukan waktu yang diperlukan satu kendaraan untuk melayani semua puskesmas ataupun durasi penyelesaian operasi dengan kendaraan berjalan paralel.

## File notebook

| Notebook | Isi |
|---|---|
| `fetch-data.ipynb` | Pengambilan koordinat, pembentukan graf jalan, node snapping, serta matriks Dijkstra |
| `kelompok-clustering.ipynb` | Perbandingan Agglomerative dan K-Medoids; evaluasi Silhouette dan Calinski-Harabasz |
| `data-augmentation.ipynb` | Simulasi 20 skenario waktu perjalanan |
| `reinforcement-learning.ipynb` | RL awal, tuning Q-Learning dengan checkpoint, dan perbandingan NN |

**Urutan menjalankan notebook:** `fetch-data` → `kelompok-clustering` → `data-augmentation` → `reinforcement-learning`.

> **Penting:** `kelompok-clustering.ipynb` masih menyertakan bagian eksperimen **`#SEBELUMNYA` (HDBSCAN)** sebagai arsip. Untuk mereproduksi metode final, **hanya jalankan bagian awal Agglomerative vs K-Medoids** dan hentikan sebelum bagian tersebut. Notebook RL juga memiliki bagian **`#TUNING LAMA`**; gunakan bagian **`#TUNING RL`** untuk reproduksi hasil utama.

## Persiapan dan cara menjalankan di Google Colab

Notebook dibuat untuk **Google Colab** dengan Google Drive. Path kerja yang digunakan pada kode adalah:

```python
folder = "/content/drive/MyDrive/puskesmas-rute"
```

1. Unggah ke GitHub keempat notebook. Jika notebook asli bernama `fetch-data(1).ipynb`, ubah menjadi `fetch-data.ipynb` agar konsisten dengan dokumentasi ini.
2. Buka notebook melalui Google Colab, hubungkan Google Drive, dan pastikan folder `MyDrive/puskesmas-rute` tersedia.
3. Instal dependensi yang diperlukan, misalnya:

   ```python
   !pip install -q numpy pandas matplotlib scikit-learn requests osmnx networkx geopandas shapely
   ```

4. Jalankan **Fetch Data** untuk membuat `puskesmas_surabaya.csv`, graf jalan, `lokasi_osm_nodes.csv`, `matrix_length.npy`, dan `matrix_time.npy`. Pengambilan lokasi dari Google Places memerlukan kredensial API yang valid dan dapat dikenakan kuota/biaya layanan.
5. Jalankan **Clustering** sampai akhir bagian eksperimen **k=11**, dan pastikan file `kelompok_agglomerative_k11.csv` terbentuk.
6. Jalankan **Data Augmentation**; proses ini membutuhkan `surabaya_routing.graphml` dan menghasilkan matriks skenario di subfolder `data_augmentation/`.
7. Jalankan **Reinforcement Learning**, khususnya bagian tuning tahap kedua, untuk memperoleh hasil validasi dan testing.

Jika menggunakan struktur direktori repository yang berbeda dari folder Drive di atas, **sesuaikan path pembacaan/penyimpanan pada notebook** sebelum menjalankan.

### Berkas masukan/keluaran penting

```text
puskesmas-rute/                      # Folder kerja di Google Drive
├── puskesmas_surabaya.csv
├── lokasi_osm_nodes.csv
├── surabaya_drive.graphml            # Graf awal (besar; opsional jika akan dibuat ulang)
├── surabaya_routing.graphml          # Graf yang digunakan untuk routing
├── surabaya_routing_fixed.graphml    # Graf setelah perbaikan travel_time
├── matrix_length.npy
├── matrix_time.npy
├── kelompok_agglomerative_k11.csv
├── kelompok_kmedoids_k11.csv
├── evaluasi_clustering_k11.csv
├── perbandingan_clustering_k11.csv
├── grafik_silhouette_k11.png
├── grafik_calinski_k11.png
├── data_augmentation/
│   ├── matrix_time_aug.npy
│   ├── urutan_lokasi_aug.csv
│   ├── skenario_traffic.csv
│   └── ringkasan_skenario.csv
└── reinforcement_learning/
    ├── tuning_rl_v2_detail_agg_k11.csv
    ├── tuning_rl_v2_ringkasan_agg_k11.csv
    ├── grafik_tuning_rl_v2_agg_k11.png
    ├── testing_rl_v2_detail_agg_k11.csv
    ├── testing_rl_v2_total_agg_k11.csv
    └── rute_rl_v2_agg_k11.csv
```

Struktur tersebut menjelaskan **lokasi berkas pada saat notebook dijalankan di Colab**, bukan kewajiban mengunggah seluruh file ke GitHub. Graf `.graphml` berukuran puluhan MB dapat disimpan di Drive atau dibuat ulang melalui notebook Fetch Data. Jangan mengunggah file besar atau duplikat eksperimen jika tidak diperlukan.

## Keterbatasan

- Skenario lalu lintas bersifat sintetis; belum divalidasi terhadap kondisi lalu lintas real-time.
- Rute belum memperhitungkan **kapasitas kendaraan, permintaan/jumlah obat, jam operasional, dan waktu bongkar muat**.
- Tuning tahap kedua yang dilaporkan menggunakan **satu random seed (42)**. Kestabilan hasil antarseed belum diuji lebih lanjut.
- **K-Medoids menggunakan implementasi pembaruan medoid bergantian**, bukan implementasi PAM penuh.
- Calinski-Harabasz dihitung pada proyeksi **MDS lima dimensi**, sehingga nilainya bukan evaluasi langsung atas matriks jarak jaringan jalan.
- Komputasi Q-Learning lebih mahal daripada NN; keunggulan waktu rute tidak sama dengan keunggulan waktu komputasi.
- Skenario S16–S19 telah digunakan dalam evaluasi eksperimen penelitian, sehingga studi berikutnya memerlukan skenario baru yang benar-benar belum pernah ditinjau untuk menguatkan klaim generalisasi.

## Sumber data dan keamanan

- **Google Places API:** referensi koordinat dan nama tempat (perhatikan ketentuan penggunaan serta pembagian ulang data).
- **OpenStreetMap:** jaringan jalan (atribusi kepada kontributor OpenStreetMap sesuai ketentuan lisensinya).
- **OSMnx dan NetworkX:** pengolahan graf jalan dan perhitungan jalur terpendek.

**Jangan unggah API key, token, kredensial, atau file rahasia ke repository publik**, termasuk output notebook yang mungkin menampilkannya. Gunakan **Colab Secrets** atau variabel lingkungan untuk menyimpan kredensial. Periksa juga izin publikasi ulang data yang diperoleh dari API.

---

*Proyek akademik Machine Learning II, studi kasus distribusi obat ke puskesmas di Kota Surabaya.*
