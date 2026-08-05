import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.*;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BiFunction;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

// =====================================================================
//  VectorDB Engine — Java port  / httplib version
// =====================================================================
public class VectorDBEngine {

    static final int DIMS = 16; // demo vectors

    // =================================================================
    //  DATA TYPES
    // =================================================================

    static class VectorItem {
        int id;
        String metadata;
        String category;
        float[] emb;

        VectorItem(int id, String metadata, String category, float[] emb) {
            this.id = id; this.metadata = metadata; this.category = category; this.emb = emb;
        }
    }

    interface DistFn extends BiFunction<float[], float[], Float> {}

    // =================================================================
    //  DISTANCE METRICS
    // =================================================================

    static float euclidean(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) { float d = a[i] - b[i]; s += d * d; }
        return (float) Math.sqrt(s);
    }

    static float cosine(float[] a, float[] b) {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / ((float) Math.sqrt(na) * (float) Math.sqrt(nb));
    }

    static float manhattan(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s;
    }

    static DistFn getDistFn(String m) {
        if ("cosine".equals(m)) return VectorDBEngine::cosine;
        if ("manhattan".equals(m)) return VectorDBEngine::manhattan;
        return VectorDBEngine::euclidean;
    }

    // =================================================================
    //  BRUTE FORCE
    // =================================================================

    static class BruteForce {
        List<VectorItem> items = new ArrayList<>();

        void insert(VectorItem v) { items.add(v); }

        List<float[]> knnRaw(float[] q, int k, DistFn dist) { return null; } // unused placeholder

        List<Map.Entry<Float, Integer>> knn(float[] q, int k, DistFn dist) {
            List<Map.Entry<Float, Integer>> r = new ArrayList<>();
            for (VectorItem v : items) r.add(Map.entry(dist.apply(q, v.emb), v.id));
            r.sort(Map.Entry.comparingByKey());
            if (r.size() > k) r = r.subList(0, k);
            return new ArrayList<>(r);
        }

        void remove(int id) {
            items.removeIf(v -> v.id == id);
        }
    }

    // =================================================================
    //  KD-TREE
    // =================================================================

    static class KDNode {
        VectorItem item;
        KDNode left, right;
        KDNode(VectorItem v) { item = v; }
    }

    static class KDTree {
        KDNode root;
        int dims;

        KDTree(int d) { dims = d; }

        KDNode ins(KDNode n, VectorItem v, int d) {
            if (n == null) return new KDNode(v);
            int ax = d % dims;
            if (v.emb[ax] < n.item.emb[ax]) n.left = ins(n.left, v, d + 1);
            else n.right = ins(n.right, v, d + 1);
            return n;
        }

        void insert(VectorItem v) { root = ins(root, v, 0); }

        void knnRec(KDNode n, float[] q, int k, int d, DistFn dist,
                    PriorityQueue<Map.Entry<Float, Integer>> heap) {
            if (n == null) return;
            float dn = dist.apply(q, n.item.emb);
            if (heap.size() < k || dn < heap.peek().getKey()) {
                heap.add(Map.entry(dn, n.item.id));
                if (heap.size() > k) heap.poll();
            }
            int ax = d % dims;
            float diff = q[ax] - n.item.emb[ax];
            KDNode closer = diff < 0 ? n.left : n.right;
            KDNode farther = diff < 0 ? n.right : n.left;
            knnRec(closer, q, k, d + 1, dist, heap);
            if (heap.size() < k || Math.abs(diff) < heap.peek().getKey())
                knnRec(farther, q, k, d + 1, dist, heap);
        }

        List<Map.Entry<Float, Integer>> knn(float[] q, int k, DistFn dist) {
            // max-heap (largest first) so we can pop the worst
            PriorityQueue<Map.Entry<Float, Integer>> heap =
                new PriorityQueue<>((a, b) -> Float.compare(b.getKey(), a.getKey()));
            knnRec(root, q, k, 0, dist, heap);
            List<Map.Entry<Float, Integer>> r = new ArrayList<>(heap);
            r.sort(Map.Entry.comparingByKey());
            return r;
        }

        void rebuild(Collection<VectorItem> items) {
            root = null;
            for (VectorItem v : items) insert(v);
        }
    }

    // =================================================================
    //  HNSW — Hierarchical Navigable Small World
    // =================================================================

    static class HNSWNode {
        VectorItem item;
        int maxLyr;
        List<List<Integer>> nbrs;
        HNSWNode(VectorItem item, int maxLyr, List<List<Integer>> nbrs) {
            this.item = item; this.maxLyr = maxLyr; this.nbrs = nbrs;
        }
    }

    static class HNSW {
        Map<Integer, HNSWNode> G = new HashMap<>();
        int M, M0, ef_build;
        double mL;
        int topLayer = -1;
        int entryPt = -1;
        Random rng = new Random(42);

        HNSW(int m, int efBuild) {
            M = m; M0 = 2 * m; ef_build = efBuild;
            mL = 1.0 / Math.log((double) m);
        }

        int randLevel() {
            double u = rng.nextDouble();
            if (u <= 0) u = 1e-12;
            return (int) Math.floor(-Math.log(u) * mL);
        }

        List<Map.Entry<Float, Integer>> searchLayer(float[] q, int ep, int ef, int lyr, DistFn dist) {
            Map<Integer, Boolean> vis = new HashMap<>();
            PriorityQueue<Map.Entry<Float, Integer>> cands =
                new PriorityQueue<>(Map.Entry.comparingByKey()); // min-heap
            PriorityQueue<Map.Entry<Float, Integer>> found =
                new PriorityQueue<>((a, b) -> Float.compare(b.getKey(), a.getKey())); // max-heap

            float d0 = dist.apply(q, G.get(ep).item.emb);
            vis.put(ep, true);
            cands.add(Map.entry(d0, ep));
            found.add(Map.entry(d0, ep));

            while (!cands.isEmpty()) {
                Map.Entry<Float, Integer> c = cands.poll();
                float cd = c.getKey(); int cid = c.getValue();
                if (found.size() >= ef && cd > found.peek().getKey()) break;
                HNSWNode node = G.get(cid);
                if (node == null || lyr >= node.nbrs.size()) continue;
                for (int nid : node.nbrs.get(lyr)) {
                    if (vis.getOrDefault(nid, false) || !G.containsKey(nid)) continue;
                    vis.put(nid, true);
                    float nd = dist.apply(q, G.get(nid).item.emb);
                    if (found.size() < ef || nd < found.peek().getKey()) {
                        cands.add(Map.entry(nd, nid));
                        found.add(Map.entry(nd, nid));
                        if (found.size() > ef) found.poll();
                    }
                }
            }

            List<Map.Entry<Float, Integer>> res = new ArrayList<>(found);
            res.sort(Map.Entry.comparingByKey());
            return res;
        }

        List<Integer> selectNbrs(List<Map.Entry<Float, Integer>> cands, int maxM) {
            List<Integer> r = new ArrayList<>();
            for (int i = 0; i < Math.min(cands.size(), maxM); i++) r.add(cands.get(i).getValue());
            return r;
        }

        void insert(VectorItem item, DistFn dist) {
            int id = item.id;
            int lvl = randLevel();
            List<List<Integer>> nbrs = new ArrayList<>();
            for (int i = 0; i <= lvl; i++) nbrs.add(new ArrayList<>());
            G.put(id, new HNSWNode(item, lvl, nbrs));

            if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

            int ep = entryPt;
            for (int lc = topLayer; lc > lvl; lc--) {
                HNSWNode epNode = G.get(ep);
                if (epNode != null && lc < epNode.nbrs.size()) {
                    List<Map.Entry<Float, Integer>> W = searchLayer(item.emb, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = W.get(0).getValue();
                }
            }
            for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
                List<Map.Entry<Float, Integer>> W = searchLayer(item.emb, ep, ef_build, lc, dist);
                int maxM = (lc == 0) ? M0 : M;
                List<Integer> sel = selectNbrs(W, maxM);
                G.get(id).nbrs.set(lc, sel);

                for (int nid : sel) {
                    HNSWNode nnode = G.get(nid);
                    if (nnode == null) continue;
                    while (nnode.nbrs.size() <= lc) nnode.nbrs.add(new ArrayList<>());
                    List<Integer> conn = nnode.nbrs.get(lc);
                    conn.add(id);
                    if (conn.size() > maxM) {
                        List<Map.Entry<Float, Integer>> ds = new ArrayList<>();
                        for (int c : conn) if (G.containsKey(c))
                            ds.add(Map.entry(dist.apply(nnode.item.emb, G.get(c).item.emb), c));
                        ds.sort(Map.Entry.comparingByKey());
                        conn.clear();
                        for (int i = 0; i < maxM && i < ds.size(); i++) conn.add(ds.get(i).getValue());
                    }
                }
                if (!W.isEmpty()) ep = W.get(0).getValue();
            }
            if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
        }

        List<Map.Entry<Float, Integer>> knn(float[] q, int k, int ef, DistFn dist) {
            if (entryPt == -1) return new ArrayList<>();
            int ep = entryPt;
            for (int lc = topLayer; lc > 0; lc--) {
                HNSWNode epNode = G.get(ep);
                if (epNode != null && lc < epNode.nbrs.size()) {
                    List<Map.Entry<Float, Integer>> W = searchLayer(q, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = W.get(0).getValue();
                }
            }
            List<Map.Entry<Float, Integer>> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
            if (W.size() > k) W = W.subList(0, k);
            return new ArrayList<>(W);
        }

        void remove(int id) {
            if (!G.containsKey(id)) return;
            for (HNSWNode nd : G.values())
                for (List<Integer> layer : nd.nbrs) layer.removeIf(x -> x == id);
            if (entryPt == id) {
                entryPt = -1;
                for (int nid : G.keySet()) if (nid != id) { entryPt = nid; break; }
            }
            G.remove(id);
        }

        static class NV { int id; String metadata, category; int maxLyr;
            NV(int id, String m, String c, int l) { this.id = id; metadata = m; category = c; maxLyr = l; } }
        static class EV { int src, dst, lyr;
            EV(int s, int d, int l) { src = s; dst = d; lyr = l; } }

        static class GraphInfo {
            int topLayer, nodeCount;
            List<Integer> nodesPerLayer = new ArrayList<>();
            List<Integer> edgesPerLayer = new ArrayList<>();
            List<NV> nodes = new ArrayList<>();
            List<EV> edges = new ArrayList<>();
        }

        GraphInfo getInfo() {
            GraphInfo gi = new GraphInfo();
            gi.topLayer = topLayer;
            gi.nodeCount = G.size();
            int maxL = Math.max(topLayer + 1, 1);
            for (int i = 0; i < maxL; i++) { gi.nodesPerLayer.add(0); gi.edgesPerLayer.add(0); }
            for (Map.Entry<Integer, HNSWNode> e : G.entrySet()) {
                int id = e.getKey(); HNSWNode nd = e.getValue();
                gi.nodes.add(new NV(id, nd.item.metadata, nd.item.category, nd.maxLyr));
                for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                    gi.nodesPerLayer.set(lc, gi.nodesPerLayer.get(lc) + 1);
                    if (lc < nd.nbrs.size())
                        for (int nid : nd.nbrs.get(lc))
                            if (id < nid) {
                                gi.edgesPerLayer.set(lc, gi.edgesPerLayer.get(lc) + 1);
                                gi.edges.add(new EV(id, nid, lc));
                            }
                }
            }
            return gi;
        }

        int size() { return G.size(); }
    }

    // =================================================================
    //  VECTOR DATABASE  (demo 16D index)
    // =================================================================

    static class VectorDB {
        Map<Integer, VectorItem> store = new HashMap<>();
        BruteForce bf = new BruteForce();
        KDTree kdt;
        HNSW hnsw;
        ReentrantLock mu = new ReentrantLock();
        int nextId = 1;
        final int dims;

        VectorDB(int d) { dims = d; kdt = new KDTree(d); hnsw = new HNSW(16, 200); }

        int insert(String meta, String cat, float[] emb, DistFn dist) {
            mu.lock();
            try {
                VectorItem v = new VectorItem(nextId++, meta, cat, emb);
                store.put(v.id, v);
                bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
                return v.id;
            } finally { mu.unlock(); }
        }

        boolean remove(int id) {
            mu.lock();
            try {
                if (!store.containsKey(id)) return false;
                store.remove(id); bf.remove(id); hnsw.remove(id);
                kdt.rebuild(store.values());
                return true;
            } finally { mu.unlock(); }
        }

        static class Hit { int id; String meta, cat; float[] emb; float dist; }
        static class SearchOut { List<Hit> hits = new ArrayList<>(); long us; String algo, metric; }

        SearchOut search(float[] q, int k, String metric, String algo) {
            mu.lock();
            try {
                DistFn dfn = getDistFn(metric);
                long t0 = System.nanoTime();

                List<Map.Entry<Float, Integer>> raw;
                if ("bruteforce".equals(algo)) raw = bf.knn(q, k, dfn);
                else if ("kdtree".equals(algo)) raw = kdt.knn(q, k, dfn);
                else raw = hnsw.knn(q, k, 50, dfn);

                long us = (System.nanoTime() - t0) / 1000;

                SearchOut out = new SearchOut(); out.us = us; out.algo = algo; out.metric = metric;
                for (Map.Entry<Float, Integer> e : raw) {
                    int id = e.getValue();
                    if (store.containsKey(id)) {
                        VectorItem v = store.get(id);
                        Hit h = new Hit();
                        h.id = id; h.meta = v.metadata; h.cat = v.category; h.emb = v.emb; h.dist = e.getKey();
                        out.hits.add(h);
                    }
                }
                return out;
            } finally { mu.unlock(); }
        }

        static class BenchOut { long bfUs, kdUs, hnswUs; int n; }

        BenchOut benchmark(float[] q, int k, String metric) {
            mu.lock();
            try {
                DistFn dfn = getDistFn(metric);
                BenchOut b = new BenchOut();
                long t;

                t = System.nanoTime(); bf.knn(q, k, dfn); b.bfUs = (System.nanoTime() - t) / 1000;
                t = System.nanoTime(); kdt.knn(q, k, dfn); b.kdUs = (System.nanoTime() - t) / 1000;
                t = System.nanoTime(); hnsw.knn(q, k, 50, dfn); b.hnswUs = (System.nanoTime() - t) / 1000;
                b.n = store.size();
                return b;
            } finally { mu.unlock(); }
        }

        List<VectorItem> all() {
            mu.lock();
            try { return new ArrayList<>(store.values()); } finally { mu.unlock(); }
        }

        HNSW.GraphInfo hnswInfo() {
            mu.lock();
            try { return hnsw.getInfo(); } finally { mu.unlock(); }
        }

        int size() {
            mu.lock();
            try { return store.size(); } finally { mu.unlock(); }
        }
    }

    // =================================================================
    //  JSON HELPERS
    // =================================================================

    static String jS(String s) {
        StringBuilder o = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            if (c == '"') o.append("\\\"");
            else if (c == '\\') o.append("\\\\");
            else if (c == '\n') o.append("\\n");
            else if (c == '\r') o.append("\\r");
            else if (c == '\t') o.append("\\t");
            else o.append(c);
        }
        return o.append('"').toString();
    }

    static String jVec(float[] v) {
        StringBuilder ss = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) ss.append(',');
            ss.append(String.format(Locale.US, "%.4f", v[i]));
        }
        return ss.append(']').toString();
    }

    static float[] parseVec(String s) {
        if (s == null || s.isEmpty()) return new float[0];
        List<Float> v = new ArrayList<>();
        for (String t : s.split(",")) {
            try { v.add(Float.parseFloat(t.trim())); } catch (Exception ignored) {}
        }
        float[] r = new float[v.size()];
        for (int i = 0; i < r.length; i++) r[i] = v.get(i);
        return r;
    }

    // Extract a JSON string field value (handles basic escape sequences)
    static String extractStr(String body, String key) {
        int p = body.indexOf('"' + key + '"');
        if (p == -1) return "";
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                char e = body.charAt(p);
                switch (e) {
                    case '"': result.append('"'); break;
                    case '\\': result.append('\\'); break;
                    case 'n': result.append('\n'); break;
                    case 'r': result.append('\r'); break;
                    case 't': result.append('\t'); break;
                    default: result.append(e); break;
                }
            } else {
                result.append(c);
            }
            p++;
        }
        return result.toString();
    }

    // Extract a JSON integer field value
    static int extractInt(String body, String key, int def) {
        int p = body.indexOf('"' + key + '"');
        if (p == -1) return def;
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        int start = p;
        while (p < body.length() && (Character.isDigit(body.charAt(p)) || body.charAt(p) == '-')) p++;
        try { return Integer.parseInt(body.substring(start, p)); } catch (Exception e) { return def; }
    }

    static class ParsedBody { String meta, cat; float[] emb; }

    static ParsedBody parseBody(String b) {
        ParsedBody pb = new ParsedBody();
        pb.meta = extractStr(b, "metadata");
        pb.cat = extractStr(b, "category");
        pb.emb = extractArr(b, "embedding");
        return pb;
    }

    static float[] extractArr(String b, String key) {
        int p = b.indexOf('"' + key + '"');
        if (p == -1) return new float[0];
        p = b.indexOf('[', p);
        if (p == -1) return new float[0];
        int e = b.indexOf(']', p);
        if (e == -1) return new float[0];
        return parseVec(b.substring(p + 1, e));
    }

    static void cors(HttpExchange ex) {
        ex.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        ex.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        ex.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");
    }

    static void sendJson(HttpExchange ex, int status, String json) throws IOException {
        ex.getResponseHeaders().add("Content-Type", "application/json");
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    static void sendHtml(HttpExchange ex, int status, String html) throws IOException {
        ex.getResponseHeaders().add("Content-Type", "text/html");
        byte[] bytes = html.getBytes(StandardCharsets.UTF_8);
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    static String readBody(HttpExchange ex) throws IOException {
        return new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
    }

    static Map<String, String> parseQuery(String query) {
        Map<String, String> m = new HashMap<>();
        if (query == null) return m;
        for (String pair : query.split("&")) {
            int idx = pair.indexOf('=');
            if (idx == -1) continue;
            try {
                String k = URLDecoder.decode(pair.substring(0, idx), StandardCharsets.UTF_8);
                String v = URLDecoder.decode(pair.substring(idx + 1), StandardCharsets.UTF_8);
                m.put(k, v);
            } catch (Exception ignored) {}
        }
        return m;
    }

    // =================================================================
    //  TEXT CHUNKER
    // =================================================================

    static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().isEmpty() ? new String[0] : text.trim().split("\\s+");
        List<String> result = new ArrayList<>();
        if (words.length == 0) return result;
        if (words.length <= chunkWords) { result.add(text); return result; }

        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder chunk = new StringBuilder();
            for (int j = i; j < end; j++) { if (j > i) chunk.append(' '); chunk.append(words[j]); }
            result.add(chunk.toString());
            if (end == words.length) break;
        }
        return result;
    }

    // =================================================================
    //  OLLAMA CLIENT — wraps local Ollama REST API
    //  Install:  https://ollama.com
    //  Models:   ollama pull nomic-embed-text
    //            ollama pull llama3.2
    // =================================================================

    static class OllamaClient {
        String host; int port;
        HttpClient client;
        String embedModel = "nomic-embed-text";
        String genModel = "llama3.2";

        OllamaClient(String h, int p) {
            host = h; port = p;
            client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
        }

        String esc(String s) {
            StringBuilder o = new StringBuilder();
            for (char c : s.toCharArray()) {
                if (c == '"') o.append("\\\"");
                else if (c == '\\') o.append("\\\\");
                else if (c == '\n') o.append("\\n");
                else if (c == '\r') o.append("\\r");
                else if (c == '\t') o.append("\\t");
                else o.append(c);
            }
            return o.toString();
        }

        float[] parseEmbedding(String body) {
            int p = body.indexOf("\"embedding\"");
            if (p == -1) return new float[0];
            p = body.indexOf('[', p);
            if (p == -1) return new float[0];
            int e = p + 1, depth = 1;
            while (e < body.length() && depth > 0) {
                char c = body.charAt(e);
                if (c == '[') depth++;
                else if (c == ']') depth--;
                e++;
            }
            return parseVec(body.substring(p + 1, e - 1));
        }

        String parseResponse(String body) { return extractStr(body, "response"); }

        boolean isAvailable() {
            try {
                HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/tags"))
                    .timeout(Duration.ofSeconds(2)).GET().build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                return res.statusCode() == 200;
            } catch (Exception e) { return false; }
        }

        float[] embed(String text) {
            try {
                String body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
                HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/embeddings"))
                    .timeout(Duration.ofSeconds(30))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) return new float[0];
                return parseEmbedding(res.body());
            } catch (Exception e) { return new float[0]; }
        }

        String generate(String prompt) {
            try {
                String body = "{\"model\":\"" + genModel + "\","
                        + "\"prompt\":\"" + esc(prompt) + "\","
                        + "\"stream\":false}";
                HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/generate"))
                    .timeout(Duration.ofSeconds(180))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) return "ERROR: Ollama unavailable. Run: ollama serve";
                return parseResponse(res.body());
            } catch (Exception e) {
                return "ERROR: Ollama unavailable. Run: ollama serve";
            }
        }
    }

    // =================================================================
    //  DOCUMENT DATABASE — HNSW over real Ollama embeddings
    // =================================================================

    static class DocItem {
        int id; String title, text; float[] emb;
        DocItem(int id, String title, String text, float[] emb) {
            this.id = id; this.title = title; this.text = text; this.emb = emb;
        }
    }

    static class DocumentDB {
        Map<Integer, DocItem> store = new HashMap<>();
        HNSW hnsw = new HNSW(16, 200);
        BruteForce bf = new BruteForce();
        ReentrantLock mu = new ReentrantLock();
        int nextId = 1;
        int dims = 0;

        int insert(String title, String text, float[] emb) {
            mu.lock();
            try {
                if (dims == 0) dims = emb.length;
                DocItem item = new DocItem(nextId++, title, text, emb);
                store.put(item.id, item);
                VectorItem vi = new VectorItem(item.id, title, "doc", emb);
                hnsw.insert(vi, VectorDBEngine::cosine);
                bf.insert(vi);
                return item.id;
            } finally { mu.unlock(); }
        }

        List<Map.Entry<Float, DocItem>> search(float[] q, int k, float maxDist) {
            mu.lock();
            try {
                if (store.isEmpty()) return new ArrayList<>();
                List<Map.Entry<Float, Integer>> raw = (store.size() < 10)
                        ? bf.knn(q, k, VectorDBEngine::cosine)
                        : hnsw.knn(q, k, 50, VectorDBEngine::cosine);
                List<Map.Entry<Float, DocItem>> out = new ArrayList<>();
                for (Map.Entry<Float, Integer> e : raw) {
                    int id = e.getValue();
                    if (store.containsKey(id) && e.getKey() <= maxDist)
                        out.add(Map.entry(e.getKey(), store.get(id)));
                }
                return out;
            } finally { mu.unlock(); }
        }

        boolean remove(int id) {
            mu.lock();
            try {
                if (!store.containsKey(id)) return false;
                store.remove(id); hnsw.remove(id); bf.remove(id);
                return true;
            } finally { mu.unlock(); }
        }

        List<DocItem> all() {
            mu.lock();
            try { return new ArrayList<>(store.values()); } finally { mu.unlock(); }
        }

        int size() {
            mu.lock();
            try { return store.size(); } finally { mu.unlock(); }
        }

        int getDims() { return dims; }
    }

    // =================================================================
    //  DEMO DATA (16D categorical vectors)
    // =================================================================

    static void loadDemo(VectorDB db) {
        DistFn dist = getDistFn("cosine");
        // Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
        db.insert("Linked List: nodes connected by pointers", "cs",
            new float[]{0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, dist);
        db.insert("Binary Search Tree: O(log n) search and insert", "cs",
            new float[]{0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, dist);
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
            new float[]{0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}, dist);
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
            new float[]{0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}, dist);
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
            new float[]{0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}, dist);
        db.insert("Calculus: derivatives integrals and limits", "math",
            new float[]{0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, dist);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
            new float[]{0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}, dist);
        db.insert("Probability: distributions random variables Bayes theorem", "math",
            new float[]{0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}, dist);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
            new float[]{0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}, dist);
        db.insert("Combinatorics: permutations combinations generating functions", "math",
            new float[]{0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}, dist);
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
            new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, dist);
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
            new float[]{0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}, dist);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
            new float[]{0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}, dist);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
            new float[]{0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}, dist);
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
            new float[]{0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}, dist);
        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
            new float[]{0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, dist);
        db.insert("Football: tackles touchdowns field goals and strategy", "sports",
            new float[]{0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}, dist);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
            new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}, dist);
        db.insert("Chess: openings endgames tactics strategic board game", "sports",
            new float[]{0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}, dist);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
            new float[]{0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}, dist);
    }

    // =================================================================
    //  HTTP SERVER
    // =================================================================

    public static void main(String[] args) throws IOException {
        VectorDB db = new VectorDB(DIMS);
        DocumentDB docDB = new DocumentDB();
        OllamaClient ollama = new OllamaClient("127.0.0.1", 11434);

        loadDemo(db);

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);

        HttpServer svr = HttpServer.create(new InetSocketAddress("0.0.0.0", 8080), 0);

        // ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────

        svr.createContext("/search", ex -> {
            if (handleOptions(ex)) return;
            cors(ex);
            Map<String, String> qp = parseQuery(ex.getRequestURI().getQuery());
            float[] q = parseVec(qp.get("v"));
            if (q.length != DIMS) {
                sendJson(ex, 200, "{\"error\":\"need " + DIMS + "D vector\"}"); return;
            }
            int k = 5;
            try { k = Integer.parseInt(qp.getOrDefault("k", "5")); } catch (Exception ignored) {}
            String metric = qp.getOrDefault("metric", "cosine"); if (metric.isEmpty()) metric = "cosine";
            String algo = qp.getOrDefault("algo", "hnsw"); if (algo.isEmpty()) algo = "hnsw";

            VectorDB.SearchOut out = db.search(q, k, metric, algo);
            StringBuilder ss = new StringBuilder();
            ss.append("{\"results\":[");
            for (int i = 0; i < out.hits.size(); i++) {
                if (i > 0) ss.append(',');
                VectorDB.Hit h = out.hits.get(i);
                ss.append("{\"id\":").append(h.id)
                  .append(",\"metadata\":").append(jS(h.meta))
                  .append(",\"category\":").append(jS(h.cat))
                  .append(",\"distance\":").append(String.format(Locale.US, "%.6f", h.dist))
                  .append(",\"embedding\":").append(jVec(h.emb)).append('}');
            }
            ss.append("],\"latencyUs\":").append(out.us)
              .append(",\"algo\":").append(jS(out.algo))
              .append(",\"metric\":").append(jS(out.metric)).append('}');
            sendJson(ex, 200, ss.toString());
        });

        svr.createContext("/insert", ex -> {
            if (handleOptions(ex)) return;
            cors(ex);
            if (!"POST".equals(ex.getRequestMethod())) { sendJson(ex, 405, "{\"error\":\"method not allowed\"}"); return; }
            String body = readBody(ex);
            ParsedBody pb = parseBody(body);
            if (pb.meta.isEmpty() || pb.emb.length != DIMS) {
                sendJson(ex, 200, "{\"error\":\"invalid body\"}"); return;
            }
            int id = db.insert(pb.meta, pb.cat, pb.emb, getDistFn("cosine"));
            sendJson(ex, 200, "{\"id\":" + id + "}");
        });

        svr.createContext("/delete/", ex -> {
            if (handleOptions(ex)) return;
            cors(ex);
            if (!"DELETE".equals(ex.getRequestMethod())) { sendJson(ex, 405, "{\"error\":\"method not allowed\"}"); return; }
            String path = ex.getRequestURI().getPath();
            Matcher m = Pattern.compile("/delete/(\\d+)").matcher(path);
            if (!m.matches()) { sendJson(ex, 404, "{\"error\":\"not found\"}"); return; }
            int id = Integer.parseInt(m.group(1));
            boolean ok = db.remove(id);
            sendJson(ex, 200, "{\"ok\":" + ok + "}");
        });

        svr.createContext("/items", ex -> {
            if (handleOptions(ex)) return;
            cors(ex);
            List<VectorItem> items = db.all();
            StringBuilder ss = new StringBuilder("[");
            for (int i = 0; i < items.size(); i++) {
                if (i > 0) ss.append(',');
                VectorItem v = items.get(i);
                ss.append("{\"id\":").append(v.id)
                  .append(",\"metadata\":").append(jS(v.metadata))
                  .append(",\"category\":").append(jS(v.category))
                  .append(",\"embedding\":").append(jVec(v.emb)).append('}');
            }
            ss.append(']');
            sendJson(ex, 200, ss.toString());
        });

        svr.createContext("/benchmark", ex -> {
            if (handleOptions(ex)) return;
            cors(ex);
            Map<String, String> qp = parseQuery(ex.getRequestURI().getQuery());
            float[] q = parseVec(qp.get("v"));
            if (q.length != DIMS) {
                sendJson(ex, 200, "{\"error\":\"need " + DIMS + "D vector\"}"); return;
            }
            int k = 5;
            try { k = Integer.parseInt(qp.getOrDefault("k", "5")); } catch (Exception ignored) {}
            String metric = qp.getOrDefault("metric", "cosine"); if (metric.isEmpty()) metric = "cosine";
            VectorDB.BenchOut b = db.benchmark(q, k, metric);
            String json = "{\"bruteforceUs\":" + b.bfUs + ",\"kdtreeUs\":" + b.kdUs
                    + ",\"hnswUs\":" + b.hnswUs + ",\"itemCount\":" + b.n + '}';
            sendJson(ex, 200, json);
        });

        svr.createContext("/hnsw-info", ex -> {
            if (handleOptions(ex)) return;
            cors(ex);
            HNSW.GraphInfo gi = db.hnswInfo();
            StringBuilder ss = new StringBuilder();
            ss.append("{\"topLayer\":").append(gi.topLayer).append(",\"nodeCount\":").append(gi.nodeCount)
              .append(",\"nodesPerLayer\":[");
            for (int i = 0; i < gi.nodesPerLayer.size(); i++) { if (i > 0) ss.append(','); ss.append(gi.nodesPerLayer.get(i)); }
            ss.append("],\"edgesPerLayer\":[");
            for (int i = 0; i < gi.edgesPerLayer.size(); i++) { if (i > 0) ss.append(','); ss.append(gi.edgesPerLayer.get(i)); }
            ss.append("],\"nodes\":[");
            for (int i = 0; i < gi.nodes.size(); i++) {
                if (i > 0) ss.append(',');
                HNSW.NV n = gi.nodes.get(i);
                ss.append("{\"id\":").append(n.id).append(",\"metadata\":").append(jS(n.metadata))
                  .append(",\"category\":").append(jS(n.category)).append(",\"maxLyr\":").append(n.maxLyr).append('}');
            }
            ss.append("],\"edges\":[");
            for (int i = 0; i < gi.edges.size(); i++) {
                if (i > 0) ss.append(',');
                HNSW.EV e = gi.edges.get(i);
                ss.append("{\"src\":").append(e.src).append(",\"dst\":").append(e.dst).append(",\"lyr\":").append(e.lyr).append('}');
            }
            ss.append("]}");
            sendJson(ex, 200, ss.toString());
        });

        // ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────

        svr.createContext("/doc/insert", ex -> {
            if (handleOptions(ex)) return;
            cors(ex);
            if (!"POST".equals(ex.getRequestMethod())) { sendJson(ex, 405, "{\"error\":\"method not allowed\"}"); return; }
            String body = readBody(ex);
            String title = extractStr(body, "title");
            String text = extractStr(body, "text");
            if (title.isEmpty() || text.isEmpty()) {
                sendJson(ex, 200, "{\"error\":\"need title and text\"}"); return;
            }

            List<String> chunks = chunkText(text, 250, 30);
            List<Integer> ids = new ArrayList<>();

            for (int i = 0; i < chunks.size(); i++) {
                float[] emb = ollama.embed(chunks.get(i));
                if (emb.length == 0) {
                    sendJson(ex, 200,
                        "{\"error\":\"Ollama unavailable. "
                        + "Install from https://ollama.com then run: "
                        + "ollama pull nomic-embed-text && ollama pull llama3.2\"}");
                    return;
                }
                String chunkTitle = (chunks.size() > 1)
                        ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                        : title;
                ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
            }

            StringBuilder ss = new StringBuilder();
            ss.append("{\"ids\":[");
            for (int i = 0; i < ids.size(); i++) { if (i > 0) ss.append(','); ss.append(ids.get(i)); }
            ss.append("],\"chunks\":").append(chunks.size())
              .append(",\"dims\":").append(docDB.getDims()).append('}');
            sendJson(ex, 200, ss.toString());
        });

        svr.createContext("/doc/delete/", ex -> {
            if (handleOptions(ex)) return;
            cors(ex);
            if (!"DELETE".equals(ex.getRequestMethod())) { sendJson(ex, 405, "{\"error\":\"method not allowed\"}"); return; }
            String path = ex.getRequestURI().getPath();
            Matcher m = Pattern.compile("/doc/delete/(\\d+)").matcher(path);
            if (!m.matches()) { sendJson(ex, 404, "{\"error\":\"not found\"}"); return; }
            int id = Integer.parseInt(m.group(1));
            boolean ok = docDB.remove(id);
            sendJson(ex, 200, "{\"ok\":" + ok + "}");
        });

        svr.createContext("/doc/list", ex -> {
            if (handleOptions(ex)) return;
            cors(ex);
            List<DocItem> docs = docDB.all();
            StringBuilder ss = new StringBuilder("[");
            for (int i = 0; i < docs.size(); i++) {
                if (i > 0) ss.append(',');
                DocItem d = docs.get(i);
                String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "…" : d.text;
                int words = 1;
                for (char c : d.text.toCharArray()) if (c == ' ') words++;
                ss.append("{\"id\":").append(d.id)
                  .append(",\"title\":").append(jS(d.title))
                  .append(",\"preview\":").append(jS(preview))
                  .append(",\"words\":").append(words).append('}');
            }
            ss.append(']');
            sendJson(ex, 200, ss.toString());
        });

        svr.createContext("/doc/search", ex -> {
            if (handleOptions(ex)) return;
            cors(ex);
            if (!"POST".equals(ex.getRequestMethod())) { sendJson(ex, 405, "{\"error\":\"method not allowed\"}"); return; }
            String body = readBody(ex);
            String question = extractStr(body, "question");
            int k = extractInt(body, "k", 3);
            if (question.isEmpty()) { sendJson(ex, 200, "{\"error\":\"need question\"}"); return; }

            float[] qEmb = ollama.embed(question);
            if (qEmb.length == 0) { sendJson(ex, 200, "{\"error\":\"Ollama unavailable\"}"); return; }

            List<Map.Entry<Float, DocItem>> hits = docDB.search(qEmb, k, 0.7f);

            StringBuilder ss = new StringBuilder();
            ss.append("{\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0) ss.append(',');
                Map.Entry<Float, DocItem> h = hits.get(i);
                ss.append("{\"id\":").append(h.getValue().id)
                  .append(",\"title\":").append(jS(h.getValue().title))
                  .append(",\"distance\":").append(String.format(Locale.US, "%.4f", h.getKey())).append('}');
            }
            ss.append("]}");
            sendJson(ex, 200, ss.toString());
        });

        svr.createContext("/doc/ask", ex -> {
            if (handleOptions(ex)) return;
            cors(ex);
            if (!"POST".equals(ex.getRequestMethod())) { sendJson(ex, 405, "{\"error\":\"method not allowed\"}"); return; }
            String body = readBody(ex);
            String question = extractStr(body, "question");
            int k = extractInt(body, "k", 3);
            if (question.isEmpty()) { sendJson(ex, 200, "{\"error\":\"need question\"}"); return; }

            float[] qEmb = ollama.embed(question);
            if (qEmb.length == 0) { sendJson(ex, 200, "{\"error\":\"Ollama unavailable\"}"); return; }

            List<Map.Entry<Float, DocItem>> hits = docDB.search(qEmb, k, 0.7f);

            StringBuilder ctx = new StringBuilder();
            for (int i = 0; i < hits.size(); i++) {
                DocItem d = hits.get(i).getValue();
                ctx.append("[").append(i + 1).append("] ").append(d.title).append(":\n")
                   .append(d.text).append("\n\n");
            }
            String prompt = "You are a helpful assistant. Answer the user's question directly. "
                    + "Use the provided context if it contains relevant information. "
                    + "If it doesn't, just use your own general knowledge. "
                    + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                    + "Just answer the question naturally.\n\n"
                    + "Context:\n" + ctx
                    + "Question: " + question + "\n\n"
                    + "Answer:";

            String answer = ollama.generate(prompt);

            StringBuilder ss = new StringBuilder();
            ss.append("{\"answer\":").append(jS(answer))
              .append(",\"model\":").append(jS(ollama.genModel))
              .append(",\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0) ss.append(',');
                Map.Entry<Float, DocItem> h = hits.get(i);
                ss.append("{\"id\":").append(h.getValue().id)
                  .append(",\"title\":").append(jS(h.getValue().title))
                  .append(",\"text\":").append(jS(h.getValue().text))
                  .append(",\"distance\":").append(String.format(Locale.US, "%.4f", h.getKey())).append('}');
            }
            ss.append("],\"docCount\":").append(docDB.size()).append('}');
            sendJson(ex, 200, ss.toString());
        });

        svr.createContext("/status", ex -> {
            if (handleOptions(ex)) return;
            cors(ex);
            boolean up = ollama.isAvailable();
            String json = "{\"ollamaAvailable\":" + up
                    + ",\"embedModel\":" + jS(ollama.embedModel)
                    + ",\"genModel\":" + jS(ollama.genModel)
                    + ",\"docCount\":" + docDB.size()
                    + ",\"docDims\":" + docDB.getDims()
                    + ",\"demoDims\":" + DIMS
                    + ",\"demoCount\":" + db.size() + '}';
            sendJson(ex, 200, json);
        });

        svr.createContext("/stats", ex -> {
            if (handleOptions(ex)) return;
            cors(ex);
            String json = "{\"count\":" + db.size()
                    + ",\"dims\":" + DIMS
                    + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
                    + ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
            sendJson(ex, 200, json);
        });

        // Serve index.html for everything else (must be registered last / most general)
        svr.createContext("/", ex -> {
            if (handleOptions(ex)) return;
            if (!"/".equals(ex.getRequestURI().getPath())) { ex.sendResponseHeaders(404, -1); return; }
            Path p = Path.of("index.html");
            if (!Files.exists(p)) { ex.sendResponseHeaders(404, -1); return; }
            String html = Files.readString(p);
            sendHtml(ex, 200, html);
        });

        svr.setExecutor(java.util.concurrent.Executors.newFixedThreadPool(16));
        svr.start();
    }

    // Returns true if this request was an OPTIONS preflight (already handled)
    static boolean handleOptions(HttpExchange ex) throws IOException {
        if ("OPTIONS".equals(ex.getRequestMethod())) {
            cors(ex);
            ex.sendResponseHeaders(204, -1);
            return true;
        }
        return false;
    }
}
