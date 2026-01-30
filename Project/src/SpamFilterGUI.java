import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.IOException;
import java.util.List;
import java.util.Collections;
import java.util.Random;

public class SpamFilterGUI extends JFrame {
    private List<CombinedSpamFilters.Ex> data;
    private List<CombinedSpamFilters.Ex> train;
    private List<CombinedSpamFilters.Ex> test;
    private JTextArea outputArea;
    private JTextField datasetField;
    
    public SpamFilterGUI() {
        setTitle("Spam Filter Analysis Tool");
        setSize(800, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        
        // Create components
        JPanel inputPanel = createInputPanel();
        outputArea = new JTextArea();
        outputArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(outputArea);
        
        // Add components to frame
        add(inputPanel, BorderLayout.NORTH);
        add(scrollPane, BorderLayout.CENTER);
        
        // Load default dataset
        loadDefaultDataset();
    }
    
    private JPanel createInputPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        
        // Dataset selection
        gbc.gridx = 0;
        gbc.gridy = 0;
        panel.add(new JLabel("Dataset Path:"), gbc);
        
        gbc.gridx = 1;
        gbc.gridy = 0;
        gbc.weightx = 1.0;
        datasetField = new JTextField("hamburger_dataset.csv", 30);
        panel.add(datasetField, gbc);
        
        gbc.gridx = 2;
        gbc.gridy = 0;
        gbc.weightx = 0.0;
        JButton loadButton = new JButton("Load Dataset");
        loadButton.addActionListener(e -> loadDataset());
        panel.add(loadButton, gbc);
        
        // Algorithm buttons
        JPanel buttonPanel = new JPanel(new GridLayout(1, 3, 10, 0));
        
        JButton nbButton = new JButton("Naïve Bayes");
        nbButton.addActionListener(e -> runNaiveBayes());
        buttonPanel.add(nbButton);
        
        JButton knnButton = new JButton("K-Nearest Neighbors");
        knnButton.addActionListener(e -> runKNN());
        buttonPanel.add(knnButton);
        
        JButton dtButton = new JButton("Decision Tree");
        dtButton.addActionListener(e -> runDecisionTree());
        buttonPanel.add(dtButton);
        
        gbc.gridx = 0;
        gbc.gridy = 1;
        gbc.gridwidth = 3;
        panel.add(buttonPanel, gbc);
        
        return panel;
    }
    
    private void loadDefaultDataset() {
        try {
            data = CombinedSpamFilters.load("hamburger_dataset.csv");
            if (data.isEmpty()) {
                outputArea.setText("Default dataset is empty or missing.");
                return;
            }
            
            Collections.shuffle(data, new Random(42));
            int split = (int)(data.size()*0.75);
            train = data.subList(0, split);
            test = data.subList(split, data.size());
            
            outputArea.setText("Loaded default dataset: hamburger_dataset.csv\n");
            outputArea.append("Total samples: " + data.size() + "\n");
            outputArea.append("Training set: " + train.size() + " samples\n");
            outputArea.append("Test set: " + test.size() + " samples\n");
        } catch (IOException ex) {
            outputArea.setText("Error loading default dataset: " + ex.getMessage());
        }
    }
    
    private void loadDataset() {
        String path = datasetField.getText().trim();
        if (path.isEmpty()) {
            path = "hamburger_dataset.csv";
            datasetField.setText(path);
        }
        
        try {
            data = CombinedSpamFilters.load(path);
            if (data.isEmpty()) {
                outputArea.setText("Dataset is empty or missing.");
                return;
            }
            
            Collections.shuffle(data, new Random(42));
            int split = (int)(data.size()*0.75);
            train = data.subList(0, split);
            test = data.subList(split, data.size());
            
            outputArea.setText("Loaded dataset: " + path + "\n");
            outputArea.append("Total samples: " + data.size() + "\n");
            outputArea.append("Training set: " + train.size() + " samples\n");
            outputArea.append("Test set: " + test.size() + " samples\n");
        } catch (IOException ex) {
            outputArea.setText("Error loading dataset: " + ex.getMessage());
        }
    }
    
    private void runNaiveBayes() {
        if (data == null || data.isEmpty()) {
            outputArea.append("\nPlease load a dataset first.\n");
            return;
        }
        
        outputArea.append("\nRunning Multinomial Naïve Bayes...\n");
        
        CombinedSpamFilters.NB nb = new CombinedSpamFilters.NB();
        nb.fit(train);
        
        // Calculate metrics
        int TP = 0, FP = 0, FN = 0, TN = 0;
        for (CombinedSpamFilters.Ex e : test) {
            int p = nb.pred(e.text());
            int y = e.y();
            if (y == 1 && p == 1) TP++;
            else if (y == 0 && p == 1) FP++;
            else if (y == 1 && p == 0) FN++;
            else TN++;
        }
        
        double acc = (TP + TN) / (double)(TP + TN + FP + FN);
        double prec = TP / (double)(TP + FP == 0 ? 1 : TP + FP);
        double rec = TP / (double)(TP + FN == 0 ? 1 : TP + FN);
        double f1 = (prec + rec == 0) ? 0 : 2 * prec * rec / (prec + rec);
        
        // Format the output exactly as requested
        outputArea.append(String.format("""
            Accuracy : %.2f %%
            Precision: %.2f %%
            Recall   : %.2f %%
            F1 score : %.2f %%
            Confusion [TP=%d FP=%d FN=%d TN=%d]%n%n""",
            100 * acc, 100 * prec, 100 * rec, 100 * f1, TP, FP, FN, TN));
    
    }
    
    private void runKNN() {
        if (data == null || data.isEmpty()) {
            outputArea.append("\nPlease load a dataset first.\n");
            return;
        }
        
        outputArea.append("\nRunning K-Nearest-Neighbors (k=5)...\n");
        
        CombinedSpamFilters.KNN knn = new CombinedSpamFilters.KNN(train);
        int correct = 0;
        for (CombinedSpamFilters.Ex e : test) {
            if (knn.pred(e.text()) == e.y()) correct++;
        }
        
        double accuracy = correct * 100.0 / test.size();
        outputArea.append(String.format("Accuracy: %.2f%% (%d out of %d)%n", 
            accuracy, correct, test.size()));
    }
    
    private void runDecisionTree() {
        if (data == null || data.isEmpty()) {
            outputArea.append("\nPlease load a dataset first.\n");
            return;
        }
        
        outputArea.append("\nRunning Simple Decision Tree...\n");
        
        CombinedSpamFilters.DT dt = new CombinedSpamFilters.DT(train);
        int correct = 0;
        for (CombinedSpamFilters.Ex e : test) {
            if (dt.pred(e.text()) == e.y()) correct++;
        }
        
        double accuracy = correct * 100.0 / test.size();
        outputArea.append(String.format("Accuracy: %.2f%% (%d out of %d)%n", 
            accuracy, correct, test.size()));
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            SpamFilterGUI gui = new SpamFilterGUI();
            gui.setVisible(true);
        });
    }
}