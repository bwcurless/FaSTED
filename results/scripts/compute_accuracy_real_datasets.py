import pathlib  
import compute_accuracy

if __name__ == "__main__":
    # Create input data
    dataset_path = "/scratch/bc2497/pairsData/"
    # Dataset 1, 2, 3, 4
    comparisons = [
	    ("cifar60k_unscaled_0.628906.txt",
	     "flattened_stripped_neighbortable_FP64_cifar60k_eps_0.62890625.txt"),
	    ("gist_unscaled_0.473633.txt",
	     "flattened_stripped_neighbortable_FP64_gist_eps_0.4736328125.txt"),
	    ("sift10m_unscaled_122.500000.txt",
	     "flattened_stripped_neighbortable_FP64_sift10m_eps_122.5.txt"),
	    ("tiny5m_unscaled_0.183105.txt",
	     "flattened_stripped_neighbortable_FP64_tiny5m_eps_0.18310546875.txt"),
    ]



    results = {}

    for (left_file, right_file) in comparisons:
        print(f"Comparing file: {left_file} with file: {right_file}")

        left_path = pathlib.Path(dataset_path, left_file)
        right_path = pathlib.Path(dataset_path, right_file)

        input_data = compute_accuracy.create_input_data_from_files(left_path, right_path) 

        pair_comparison = compute_accuracy.compute_pair_stats(input_data)

        results[f"{left_file}, {right_file}"] = pair_comparison

        print("Comparison done")

    print("Final comparison results")
    print(results)
