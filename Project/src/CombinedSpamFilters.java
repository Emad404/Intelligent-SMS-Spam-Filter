
import java.io.*;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

 
public class CombinedSpamFilters {

    private static final Set<String> STOP = loadStopWords("stopwords_en.txt");
    private static Set<String> loadStopWords(String f) {
        try { return new HashSet<>(Files.readAllLines(Paths.get(f))); }
        catch (IOException e) { System.err.println("Stop-word file not found."); return Set.of(); }
    }

    private static List<String> tok(String s) {
        String[] parts = s.toLowerCase().replaceAll("[^a-z\\s]", " ").split("\\s+");
        List<String> out = new ArrayList<>();
        for (String w : parts) if (!w.isBlank() && !STOP.contains(w)) out.add(w);
        return out;
    }

    record Ex(String text, int y) {}

    static List<Ex> load(String src) throws IOException {
        BufferedReader br = src.startsWith("http")
                ? new BufferedReader(new InputStreamReader(new URL(src).openStream()))
                : Files.newBufferedReader(Paths.get(src));

        List<Ex> list = new ArrayList<>();
        for (String line; (line = br.readLine()) != null;) {
            String[] p = line.split("\t|,", 2);          
            if (p.length < 2) continue;
            int y = p[0].equalsIgnoreCase("spam") || p[0].equals("1") ? 1 : 0;
            list.add(new Ex(p[1], y));
        }
        br.close();
        return list;
    }

    static final class NB {
        Map<String,Integer> id = new HashMap<>();
        int V; double a = 1;
        double[] logPrior = new double[2];
        double[][] logLik;

        void fit(List<Ex> ds) {
            long[] doc = new long[2], tokCnt = new long[2];
            Map<Integer,long[]> cnt = new HashMap<>();

            for (Ex e : ds) {
                doc[e.y()]++;
                cnt.computeIfAbsent(e.y(), k -> new long[10000]);
                for (String t : tok(e.text())) {
                    int v = id.computeIfAbsent(t, k -> V++);
                    long[] arr = cnt.get(e.y());
                    if (v >= arr.length) arr = Arrays.copyOf(arr, v * 2);
                    arr[v]++; cnt.put(e.y(), arr); tokCnt[e.y()]++;
                }
            }
            logLik = new double[2][V];
            for (int c = 0; c < 2; c++) {
                long[] arr = cnt.getOrDefault(c, new long[V]);
                if (arr.length < V) arr = Arrays.copyOf(arr, V);
                for (int v = 0; v < V; v++)
                    logLik[c][v] = Math.log(arr[v] + a) - Math.log(tokCnt[c] + a * V);
            }
            long tot = doc[0] + doc[1];
            logPrior[0] = Math.log((double) doc[0] / tot);
            logPrior[1] = Math.log((double) doc[1] / tot);
        }
        int pred(String txt) {
            double[] s = { logPrior[0], logPrior[1] };
            for (String t : tok(txt)) {
                Integer v = id.get(t);
                if (v != null) { s[0] += logLik[0][v]; s[1] += logLik[1][v]; }
            }
            return s[1] > s[0] ? 1 : 0;
        }
    }

    static final class KNN {
        record Vec(Map<String,Integer> f,int y){}
        List<Vec> train; int k = 5; Set<String> vocab;

        KNN(List<Ex> tr) {
            vocab = new HashSet<>();
            for (Ex e : tr) vocab.addAll(tok(e.text()));
            train = new ArrayList<>();
            for (Ex e : tr) train.add(new Vec(tf(e.text()), e.y()));
        }
        Map<String,Integer> tf(String t) {
            Map<String,Integer> m = new HashMap<>();
            for (String w : tok(t)) if (vocab.contains(w)) m.put(w, m.getOrDefault(w,0)+1);
            return m;
        }
        int pred(String txt) {
            Map<String,Integer> q = tf(txt);
            List<Map.Entry<Integer,Double>> dist = new ArrayList<>();
            for (int i = 0; i < train.size(); i++)
                dist.add(Map.entry(i, euclid(q, train.get(i).f)));
            dist.sort(Comparator.comparingDouble(Map.Entry::getValue));
            int spam = 0;
            for (int i = 0; i < k; i++) if (train.get(dist.get(i).getKey()).y == 1) spam++;
            return spam > k/2 ? 1 : 0;
        }
        double euclid(Map<String,Integer> a, Map<String,Integer> b) {
            Set<String> keys = new HashSet<>(a.keySet()); keys.addAll(b.keySet());
            double sum = 0;
            for (String k : keys)
                sum += Math.pow(a.getOrDefault(k,0) - b.getOrDefault(k,0), 2);
            return Math.sqrt(sum);
        }
    }

    static final class DT {
        private static final String[] WORDS =
                {"free","win","money","urgent","click","reward","call"};
        private record Node(String w,String lab,Node yes,Node no){}
        private Node root;
        DT(List<Ex> tr) { root = build(tr, new LinkedHashSet<>(Arrays.asList(WORDS))); }
        int pred(String t) {
            Node n = root; Set<String> set = new HashSet<>(tok(t));
            while (n.lab == null) n = set.contains(n.w) ? n.yes : n.no;
            return n.lab.equals("spam") ? 1 : 0;
        }
        private Node build(List<Ex> d, Set<String> attrs) {
            int pos = (int) d.stream().filter(e -> e.y() == 1).count();
            if (pos == 0 || pos == d.size())
                return new Node(null, pos == 0 ? "ham" : "spam", null, null);
            if (attrs.isEmpty())
                return new Node(null, pos >= d.size()/2.0 ? "spam" : "ham", null, null);
            String best = attrs.iterator().next();          // pick first word
            List<Ex> yes = new ArrayList<>(), no = new ArrayList<>();
            for (Ex e : d) (tok(e.text()).contains(best) ? yes : no).add(e);
            Set<String> rest = new LinkedHashSet<>(attrs); rest.remove(best);
            return new Node(best, null, build(yes, rest), build(no, rest));
        }
    }

    static void eval(List<Ex> test, java.util.function.Function<String,Integer> f) {
        int TP=0,FP=0,FN=0,TN=0;
        for (Ex e : test) {
            int p = f.apply(e.text()), y = e.y();
            if (y==1 && p==1) TP++; else if (y==0 && p==1) FP++;
            else if (y==1 && p==0) FN++; else TN++;
        }
        double acc  = (TP+TN)/(double)(TP+TN+FP+FN);
        double prec = TP/(double)(TP+FP==0?1:TP+FP);
        double rec  = TP/(double)(TP+FN==0?1:TP+FN);
        double f1   = (prec+rec==0)?0:2*prec*rec/(prec+rec);
        System.out.printf("""
                Accuracy : %.2f %%
                Precision: %.2f %%
                Recall   : %.2f %%
                F1 score : %.2f %%
                Confusion [TP=%d FP=%d FN=%d TN=%d]%n%n""",
                100*acc, 100*prec, 100*rec, 100*f1, TP,FP,FN,TN);
    }

    public static void main(String[] args) throws Exception {
        Scanner sc = new Scanner(System.in);

        String path = (args.length == 1) ? args[0] : "hamburger_dataset.csv";
        List<Ex> data = load(path);
        if (data.isEmpty()) { System.out.println("Dataset is empty or missing."); return; }

        Collections.shuffle(data, new Random(42));
        int split = (int)(data.size()*0.75);
        List<Ex> train = data.subList(0, split);
        List<Ex> test  = data.subList(split, data.size());

        while (true) {
            System.out.println("""
                    Choose algorithm:
                    1 – Multinomial Naïve Bayes
                    2 – K-Nearest-Neighbours
                    3 – Simple Decision Tree
                    4 – Exit""");
            System.out.print("> ");
            int choice = sc.nextInt(); sc.nextLine();

            switch (choice) {
                case 1 -> { NB nb = new NB(); nb.fit(train);  eval(test, nb::pred); }
                case 2 -> { KNN knn = new KNN(train);
                            int correct=0; for(Ex e:test) if(knn.pred(e.text())==e.y()) correct++;
                            System.out.printf("Accuracy: %.2f%% (%d out of %d)%n%n",
                                              correct*100.0/test.size(), correct, test.size()); }
                case 3 -> { DT dt = new DT(train);
                            int correct=0; for(Ex e:test) if(dt.pred(e.text())==e.y()) correct++;
                            System.out.printf("Accuracy: %.2f%% (%d out of %d)%n%n",
                                              correct*100.0/test.size(), correct, test.size()); }
                case 4 -> { System.out.println("Good-bye!"); return; }
                default -> System.out.println("Please choose 1, 2, 3, or 4.");
            }
        }
    }

	

}
