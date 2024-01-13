import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class NormalizeValues {
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.err.println("Wrong usage!");
            System.err.println("\tUsage: java NormalizeValues <iris_path: str> <letter_recognition_path: str>");
            System.exit(1);
        }

        String irisPath = args[0].strip();
        String letterRecognitionPath = args[1].strip();

        NormalizeValues.normalizeCSV(
                Paths.get(irisPath),
                Paths.get(System.getProperty("user.dir") + File.separator + "normalized-iris.csv"));

        NormalizeValues.normalizeCSV(
                Paths.get(letterRecognitionPath),
                Paths.get(System.getProperty("user.dir") + File.separator + "normalized-letter-recognition.csv"));
    }

    /**
     * Normalizes the csv attributes to the [0,1] range.
     * @param csvPath the path of the original csv
     * @param normalizedCsvPath the path of the normalized csv
     * @return true if successful
     * <br>    false if not
     */
    @SuppressWarnings("unchecked")
    public static boolean normalizeCSV(Path csvPath, Path normalizedCsvPath) throws IOException {
        // Read head
        var bufferedReader = Files.newBufferedReader(csvPath);
        String line = bufferedReader.readLine();
        String[] attributes = line.split(",");

        // Add 1 to the class attribute, so it stays at the full right column when sorted by TreeMap
        for (int i = 0; i < attributes.length; i++) {
            if (i == attributes.length - 1)
                attributes[i] = "1" + attributes[i];
            else
                attributes[i] = "0" + attributes[i];
        }

        // Each key (attribute) links to a list of values
        // We calculate the normalized data easier this way
        var attributeValMap = new TreeMap<String, List<Object>>();

        // Read attribute values
        while ((line = bufferedReader.readLine()) != null) {
            String[] values = line.split(",");

            // Add double values
            for (int i = 0; i < values.length-1; i++) {
                List<Double> attributeValues = (List<Double>) (Object) attributeValMap.get(attributes[i]);
                if (attributeValues == null)
                    attributeValues = new ArrayList<>();

                attributeValues.add(Double.parseDouble(values[i]));
                attributeValMap.put(attributes[i], (List<Object>) (Object) attributeValues);
            }

            // Add class value separately (not a double)
            int classIndex = attributes.length - 1;
            List<String> classValues = (List<String>) (Object) attributeValMap.get(attributes[classIndex]);
            if (classValues == null)
                classValues = new ArrayList<>();

            if (values.length > 1) {
                classValues.add(values[classIndex]);
                attributeValMap.put(attributes[classIndex], (List<Object>) (Object) classValues);
            }
        }

        bufferedReader.close();


        // Normalize data for each attribute key (excluding key)
        for (int i = 0; i < attributes.length - 1; i++) {
            List<Double> values = (List<Double>) (Object) attributeValMap.get(attributes[i]);

            double min = Collections.min(values);
            double max = Collections.max(values);
            values.replaceAll(val -> (val - min) / (max - min));
        }


        // Write the values to the updated csv
        // Write head
        String comma = "";
        StringBuilder concatAttributes = new StringBuilder();

        var bufferedWriter = Files.newBufferedWriter(normalizedCsvPath);
        for (var entry : attributeValMap.entrySet()) {
            bufferedWriter.append(comma).append(entry.getKey().substring(1));
            comma = ",";
        }
        bufferedWriter.write(concatAttributes.toString());
        bufferedWriter.newLine();

        // Write values
        // Get length of a random list (all have the same size)
        long length = attributeValMap.get(attributes[0]).size();
        for (int i = 0; i < length; i++) {
            comma = "";
            for (var entry : attributeValMap.entrySet()) {
                bufferedWriter.write(comma + entry.getValue().get(i));
                comma = ",";
            }
            bufferedWriter.newLine();
        }

        bufferedWriter.close();

        return true;
    }
}