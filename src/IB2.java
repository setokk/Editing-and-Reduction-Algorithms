import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class IB2 {
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.err.println("Wrong usage!");
            System.err.println("\tUsage: java IB2 <normalizedCsvPath: str> <reducedCsvPath: str>");
            System.exit(1);
        }

        // Parameters
        Path normalizedCsvPath = Paths.get(args[0].strip());
        Path reducedCsvPath     = Paths.get(args[1].strip());

        // Read csv
        var bufferedReader = Files.newBufferedReader(normalizedCsvPath);
        String line = bufferedReader.readLine();
        String[] attributes = line.split(",");

        List<Datapoint> datapoints = new ArrayList<>();
        while ((line = bufferedReader.readLine()) != null) {
            String[] values = line.split(",");

            ArrayList<Double> datapointValues = new ArrayList<>();
            for (int i = 0; i < values.length - 1; i++) {
                datapointValues.add(Double.parseDouble(values[i]));
            }
            String classLabel = values[values.length - 1];

            datapoints.add(new Datapoint(datapointValues, classLabel));
        }

        bufferedReader.close();

        // Add first element to reduced data points (random)
        List<Datapoint> reducedDatapoints = new ArrayList<>();
        reducedDatapoints.add(datapoints.get(new Random().nextInt(datapoints.size())));

        // Apply the IB2 algorithm
        int originalSize = datapoints.size();
        Iterator<Datapoint> iterator = datapoints.listIterator();
        while (iterator.hasNext()) {
            Datapoint datapoint = iterator.next();

            DatapointNeighbor nn = findNearestNeighbor(datapoint, reducedDatapoints);
            if (!datapoint.classLabel.equals(nn.getClassLabel()))
                reducedDatapoints.add(datapoint);

            iterator.remove();
        }

        System.out.println("Original datapoints size: " + originalSize);
        System.out.println("Reduced datapoints size: " + reducedDatapoints.size());

        // Write reduced data points to csv
        var bufferedWriter = Files.newBufferedWriter(reducedCsvPath);

        // Write attributes names
        StringBuilder concatAttributes = new StringBuilder();
        String comma = "";
        for (String attribute : attributes) {
            concatAttributes.append(comma).append(attribute);
            comma = ",";
        }
        bufferedWriter.write(concatAttributes.toString());
        bufferedWriter.newLine();

        // Write actual data of attributes
        for (Datapoint datapoint : reducedDatapoints) {
            comma = "";
            StringBuilder concatValues = new StringBuilder();
            for (Double value : datapoint.values) {
                concatValues.append(comma).append(value);
                comma = ",";
            }
            // Append the class value of datapoint as well
            concatValues.append(comma).append(datapoint.classLabel);
            bufferedWriter.write(concatValues.toString());
            bufferedWriter.newLine();
        }

        bufferedWriter.close();
    }

    private static DatapointNeighbor findNearestNeighbor(Datapoint datapoint,
                                                 List<Datapoint> reducedDatapoints) {
        DatapointNeighbor nn = new DatapointNeighbor(new Datapoint(null, ""), 1_000_000);

        for (int i = 0; i < reducedDatapoints.size(); i++) {
            Datapoint oDatapoint = reducedDatapoints.get(i);
            double distance = datapoint.distanceFrom(oDatapoint);
            if (distance < nn.distance) {
                nn = new DatapointNeighbor(oDatapoint, distance);
            }
        }

        return nn;
    }

    private record Datapoint(ArrayList<Double> values, String classLabel) {
        public Double distanceFrom(Datapoint oDatapoint) {
            ArrayList<Double> oValues = oDatapoint.values();
            double sum = 0.0;
            for (int i = 0; i < oValues.size(); i++) {
                sum += Math.pow((oValues.get(i) - this.values.get(i)), 2);
            }
            return Math.sqrt(sum);
        }
    }

    private record DatapointNeighbor(Datapoint datapoint, double distance) {
        public String getClassLabel() {
            return datapoint.classLabel();
        }
    }
}
