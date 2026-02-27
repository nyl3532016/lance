/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance;

import org.lance.ipc.Query;
import org.lance.ipc.ScanOptions;

import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

// Creates a dataset with 5 batches where each batch has 80 rows
//
// The dataset has the following columns:
//
//  i   - i32      : [0, 1, ..., 399]
//  s   - &str     : ["s-0", "s-1", ..., "s-399"]
//  vec - [f32; 32]: [[0, 1, ... 31], [32, ..., 63], ... [..., (80 * 5 * 32) - 1]]
//
// An IVF-PQ index with 2 partitions is trained on this data
public class VectorSearchTest {

  // TODO: fix in https://github.com/lancedb/lance/issues/2956
  // Path tempDir= Paths.get("/Users/nn/cmit/lance/java/src/test/testdata");

  @TempDir Path tempDir;

  @Test
  void test_create_index() throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("test_create_index"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        testVectorDataset.createIndex(dataset);
        List<String> indexes = dataset.listIndexes();
        assertEquals(1, indexes.size());
        assertEquals(TestVectorDataset.indexName, indexes.get(0));
      }
    }
  }

  // rust/lance-linalg/src/distance/l2.rs:256:5:
  // 5assertion `left == right` failed
  // Directly panic instead of throwing an exception
  @Test
  void search_invalid_vector() throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("test_create_index"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        float[] key = new float[30];
        for (int i = 0; i < 30; i++) {
          key[i] = (float) (i + 30);
        }
        ScanOptions options =
            new ScanOptions.Builder()
                .nearest(
                    new Query.Builder()
                        .setColumn(TestVectorDataset.vectorColumnName)
                        .setKey(key)
                        .setK(5)
                        .setUseIndex(false)
                        .build())
                .build();
        assertThrows(
            IllegalArgumentException.class,
            () -> {
              try (Scanner scanner = dataset.newScan(options)) {
                try (ArrowReader reader = scanner.scanBatches()) {}
              }
            });
      }
    }
  }

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void test_knn(boolean createVectorIndex) throws Exception {
    try (TestVectorDataset testVectorDataset = new TestVectorDataset(tempDir.resolve("test_knn"))) {
      try (Dataset dataset = testVectorDataset.create()) {

        if (createVectorIndex) {
          testVectorDataset.createIndex(dataset);
        }
        float[] key = new float[32];
        for (int i = 0; i < 32; i++) {
          key[i] = (float) (i + 32);
        }
        List<Integer> fragments = new ArrayList<>();
        fragments.add(0);
        fragments.add(1);
        fragments.add(2);
        fragments.add(3);
        fragments.add(4);

        ScanOptions options =
            new ScanOptions.Builder()
                .prefilter(true)
                .fragmentIds(fragments)
                .nearest(
                    new Query.Builder()
                        .setColumn(TestVectorDataset.vectorColumnName)
                        .setKey(key)
                        .setK(3)
                        .setUseIndex(false)
                        .build())
                .build();
        try (Scanner scanner = dataset.newScan(options)) {
          try (ArrowReader reader = scanner.scanBatches()) {
            System.out.println("reader class is " + reader.getClass());
            System.out.println("scanner class is " + scanner.getClass());

            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            System.out.println("Schema:");
            assertTrue(reader.loadNextBatch(), "Expected at least one batch");

           // assertEquals(5, root.getRowCount(), "Expected 5 results");

            assertEquals(4, root.getSchema().getFields().size(), "Expected 4 columns");
            assertEquals("i", root.getSchema().getFields().get(0).getName());
            assertEquals("s", root.getSchema().getFields().get(1).getName());
            assertEquals(
                TestVectorDataset.vectorColumnName, root.getSchema().getFields().get(2).getName());
            assertEquals("_distance", root.getSchema().getFields().get(3).getName());

            IntVector iVector = (IntVector) root.getVector("i");
            Set<Integer> expectedI = new HashSet<>(Arrays.asList(1, 81, 161, 241, 321));
            Set<Integer> actualI = new HashSet<>();
            for (int i = 0; i < iVector.getValueCount(); i++) {
              actualI.add(iVector.get(i));
            }
            assertEquals(expectedI, actualI, "Unexpected values in 'i' column");

            Float4Vector distanceVector = (Float4Vector) root.getVector("_distance");
            float prevDistance = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < distanceVector.getValueCount(); i++) {
              float distance = distanceVector.get(i);
              assertTrue(distance >= prevDistance, "Distances should be in ascending order");
              prevDistance = distance;
            }

            assertFalse(reader.loadNextBatch(), "Expected only one batch");
          }
        }
      }
    }
  }

  @Test
  void test_knn_with_new_data() throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("test_knn_with_new_data"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        testVectorDataset.createIndex(dataset);
      }

      float[] key = new float[32];
      Arrays.fill(key, 0.0f);
      // Set k larger than the number of new rows
      int k = 20;

      List<TestCase> cases = new ArrayList<>();
      List<Optional<String>> filters = Arrays.asList(Optional.empty(), Optional.of("i > 100"));
      List<Optional<Integer>> limits = Arrays.asList(Optional.empty(), Optional.of(10));

      for (Optional<String> filter : filters) {
        for (Optional<Integer> limit : limits) {
          for (boolean useIndex : new boolean[] {true, false}) {
            cases.add(new TestCase(filter, limit, useIndex));
          }
        }
      }

      // Validate all cases
      try (Dataset dataset = testVectorDataset.appendNewData()) {
        for (TestCase testCase : cases) {
          ScanOptions.Builder optionsBuilder =
              new ScanOptions.Builder()
                  .nearest(
                      new Query.Builder()
                          .setColumn(TestVectorDataset.vectorColumnName)
                          .setKey(key)
                          .setK(k)
                          .setUseIndex(testCase.useIndex)
                          .build());

          testCase.filter.ifPresent(optionsBuilder::filter);
          testCase.limit.ifPresent(optionsBuilder::limit);

          ScanOptions options = optionsBuilder.build();

          try (Scanner scanner = dataset.newScan(options)) {
            try (ArrowReader reader = scanner.scanBatches()) {
              VectorSchemaRoot root = reader.getVectorSchemaRoot();
              assertTrue(reader.loadNextBatch(), "Expected at least one batch");

              if (testCase.filter.isPresent()) {
                int resultRows = root.getRowCount();
                int expectedRows = testCase.limit.orElse(k);
                assertTrue(
                    resultRows <= expectedRows,
                    "Expected less than or equal to " + expectedRows + " rows, got " + resultRows);
              } else {
                assertEquals(
                    testCase.limit.orElse(k), root.getRowCount(), "Unexpected number of rows");
              }

              // Top one should be the first value of new data
              IntVector iVector = (IntVector) root.getVector("i");
              assertEquals(
                  400, iVector.get(0), "First result should be the first value of new data");

              // Check if distances are in ascending order
              Float4Vector distanceVector = (Float4Vector) root.getVector("_distance");
              float prevDistance = Float.NEGATIVE_INFINITY;
              for (int i = 0; i < distanceVector.getValueCount(); i++) {
                float distance = distanceVector.get(i);
                assertTrue(distance >= prevDistance, "Distances should be in ascending order");
                prevDistance = distance;
              }

              assertFalse(reader.loadNextBatch(), "Expected only one batch");
            }
          }
        }
      }
    }
  }

  @Test
  void test_knn_with_fragments() throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("test_knn_with_fragments"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        List<Integer> fragmentIds = new ArrayList<>();
        // Assuming there are fragments, otherwise create some
        // testVectorDataset.create() creates 5 batches of 80 rows
        // We can get fragments from dataset
        // java.util.List<org.lance.Fragment> fragments = dataset.getFragments();
        // for (org.lance.Fragment f : fragments) {
        // fragmentIds.add(f.getId());
        // }
        // For simplicity, just use a dummy ID or try to get actual fragments if
        // possible
        // But getFragments returns Fragment objects which we might not have easy access
        // to ID without public API?
        // Let's check Dataset API for getFragments return type.
        // It returns List<Fragment>. Fragment has getId().
        // Let's assume we can get IDs.
        List<Fragment> fragments = dataset.getFragments();
        for (Fragment f : fragments) {
          fragmentIds.add(f.getId());
        }

        float[] key = new float[32];
        for (int i = 0; i < 32; i++) {
          key[i] = (float) (i + 32);
        }

        ScanOptions options =
            new ScanOptions.Builder()
                .fragmentIds(fragmentIds)
                .prefilter(true) // Enable prefilter!
                .nearest(
                    new Query.Builder()
                        .setColumn(TestVectorDataset.vectorColumnName)
                        .setKey(key)
                        .setK(5)
                        .build())
                .build();

        try (Scanner scanner = dataset.newScan(options)) {
          try (ArrowReader reader = scanner.scanBatches()) {
            assertTrue(reader.loadNextBatch());
            assertEquals(5, reader.getVectorSchemaRoot().getRowCount());
          }
        }
      }
    }
  }

  private static class TestCase {
    final Optional<String> filter;
    final Optional<Integer> limit;
    final boolean useIndex;

    TestCase(Optional<String> filter, Optional<Integer> limit, boolean useIndex) {
      this.filter = filter;
      this.limit = limit;
      this.useIndex = useIndex;
    }
  }
}
