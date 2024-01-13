import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;

public class ENN {
    public static void main(String[] args)
            throws IOException, InterruptedException, ExecutionException {
        if (args.length != 3) {
            System.err.println("Wrong usage!");
            System.err.println("\tUsage: java ENN <k: int> <normalizedCsvPath: str> <editedCsvPath: str>");
            System.exit(1);
        }

        // Parameters
        int k                  = Integer.parseInt(args[0]);
        Path normalizedCsvPath = Paths.get(args[1].strip());
        Path editedCsvPath     = Paths.get(args[2].strip());

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

        // Use multithreading for ENN
        int processors = Runtime.getRuntime().availableProcessors();
        int block = datapoints.size() / processors;
        int remainder = datapoints.size() % processors;

        // Loop for processors - 1 threads so that remainder gets added
        // to the last one (in case of odd size number of dataset).
        ExecutorService executor = Executors.newFixedThreadPool(processors);
        List<Future<List<Datapoint>>> futures = new ArrayList<>(processors);

        long startTime = System.currentTimeMillis();
        for (int i = 0; i < processors - 1; i++) {
            int start = i * block;
            int end = start + block;

            futures.add(executor.submit(new PartialENN(k, start, end, datapoints)));
        }

        // Add last callable
        int last = processors - 1;
        futures.add(executor.submit(new PartialENN(k, last * block, datapoints.size(), datapoints)));

        // LinkedList for efficient adding
        LinkedList<Datapoint> editedDatapoints = new LinkedList<>(datapoints);
        for (Future<List<Datapoint>> future : futures) {
            List<Datapoint> removedDatapoints = future.get();
            editedDatapoints.removeAll(removedDatapoints);
        }

        long endTime = System.currentTimeMillis();
        System.out.println("ENN finished in " + (endTime - startTime) + "ms");
        executor.close();

        System.out.println("Original datapoints size: " + datapoints.size());
        System.out.println("Edited datapoints size: " + editedDatapoints.size());

        // Write to output
        String comma = "";
        StringBuilder concatAttributes = new StringBuilder();

        // Write attribute names
        var bufferedWriter = Files.newBufferedWriter(editedCsvPath);
        for (String attribute : attributes) {
            concatAttributes.append(comma).append(attribute);
            comma = ",";
        }
        bufferedWriter.write(concatAttributes.toString());
        bufferedWriter.newLine();

        // Write values
        for (Datapoint datapoint : editedDatapoints) {
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

    private static class PartialENN implements Callable<List<Datapoint>> {

        private final int k;
        private final int start;
        private final int end;
        private final List<Datapoint> datapoints;

        public PartialENN(final int k,
                          final int start,
                          final int end,
                          final List<Datapoint> datapoints) {
            this.k = k;
            this.start = start;
            this.end = end;
            this.datapoints = datapoints;
        }

        @Override
        public List<Datapoint> call() {
            List<Datapoint> removedDatapoints = new ArrayList<>();
            List<Datapoint> datapointsWithoutX = new ArrayList<>(datapoints);
            for (int i = start; i < end; i++) {
                Datapoint datapoint = datapoints.get(i);

                // TS - {X} (optimization)
                datapointsWithoutX.set(i, null);

                // Find nearest neighbors using Euclidean Distance
                DatapointNeighbor[] nns = new DatapointNeighbor[k];
                for (Datapoint oDatapoint : datapointsWithoutX) {
                    // Check if datapoint is X
                    if (oDatapoint == null)
                        continue;

                    double distance = datapoint.distanceFrom(oDatapoint);
                    int index = containsNull(nns);
                    if (index != -1) {
                        nns[index] = new DatapointNeighbor(oDatapoint, distance);
                    }
                    else {
                        Arrays.sort(nns);
                        for (int j = 0; j < k; j++) {
                            if (distance < nns[j].distance()) {
                                nns[j] = new DatapointNeighbor(oDatapoint, distance);
                                break;
                            }
                        }
                    }
                }

                // Add {X} back
                datapointsWithoutX.set(i, datapoint);

                // Find major class between k nearest neighbors
                String majorClass = findMajorClass(nns);

                // Check if datapoint is not in major class
                if (!datapoint.classLabel().equals(majorClass))
                    removedDatapoints.add(datapoint);
            }

            return removedDatapoints;
        }
    }

    /**
     * Used for representing a single data point of the training set.
     * It assumes that every value is numeric.
     * It also assumes that there is an existing class label for the data point.
     * @param values List containing the attribute values (size = numOfAttributes-1)
     * @param classLabel The class label of the datapoint
     */
    private record Datapoint(ArrayList<Double> values, String classLabel) {

        /**
         * Calculates the Euclidean distance between this point and another point.
         * @param oDatapoint The other point.
         * @return The Euclidean distance.
         */
        public Double distanceFrom(Datapoint oDatapoint) {
                ArrayList<Double> oValues = oDatapoint.values();
                double sum = 0.0;
                for (int i = 0; i < oValues.size(); i++) {
                    sum += Math.pow((oValues.get(i) - this.values.get(i)), 2);
                }
                return Math.sqrt(sum);
        }
    }

    /**
     * Used in the process of finding the KNN for a specific datapoint.
     * @param datapoint A datapoint.
     * @param distance The distance this datapoint has from the target.
     */
    private record DatapointNeighbor(Datapoint datapoint, double distance) implements Comparable<DatapointNeighbor> {
        public String getClassLabel() {
                return datapoint.classLabel();
        }

        @Override
        public int compareTo(DatapointNeighbor datapointNeighbor) {
            double diff = this.distance - datapointNeighbor.distance();
            if (diff > 0)
                return 1;
            else if (diff < 0)
                return -1;
            else
                return 0;
        }
    }

    /**
     * Find the major class in a given array of nearest neighbors.
     * It assumes that k is an odd number.
     * @param nns Array containing the nearest neighbors.
     * @return The string of the major class.
     */
    private static String findMajorClass(DatapointNeighbor[] nns) {
        TreeMap<String, Integer> classesCount = new TreeMap<>();
        for (DatapointNeighbor nn : nns) {
            String classLabel = nn.getClassLabel();
            if (!classesCount.containsKey(classLabel)) {
                classesCount.put(classLabel, 1);
            } else {
                int count = classesCount.get(classLabel);
                classesCount.put(classLabel, ++count);
            }
        }

        // Find major class
        int maxCount = -1;
        String majorClass = "";
        for (Map.Entry<String, Integer> entry : classesCount.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                majorClass = entry.getKey();
            }
        }

        return majorClass;
    }

    /**
     * Checks whether the provided array has any null values
     * and returns the index of the first null value it encounters.
     * @param nns The provided array.
     * @return The index of the null value that is the most close to the start of the array.
     * Returns -1 if no null value is found.
     */
    private static int containsNull(DatapointNeighbor[] nns) {
        for (int i = 0; i < nns.length; i++) {
            if (nns[i] == null)
                return i;
        }
        return -1;
    }
}
