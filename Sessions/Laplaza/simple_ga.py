import random

from rdkit import Chem  # Ensure Chem is imported
from rdkit.Chem import QED, Crippen, Descriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib.pyplot as plt
from IPython.display import Image, SVG, display

_HAS_CAIRO = hasattr(rdMolDraw2D, "MolDraw2DCairo")


def _mol_grid_image(mols, mols_per_row, sub_img_size, legends=None):
    """Return a displayable molecule grid (PIL or SVG depending on RDKit build)."""
    kwargs = dict(
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        useSVG=not _HAS_CAIRO,
    )
    if legends is not None:
        kwargs["legends"] = legends
    return Draw.MolsToGridImage(mols, **kwargs)


def _display_mol_grid(img):
    if isinstance(img, str):
        display(SVG(img))
    else:
        display(img)


# Define a basic set of characters that can appear in SMILES
# This is a simplification; real SMILES alphabets are more complex
# and depend on the chemical space being explored.
# User can extend this list based on their need, e.g.
# SMILES_CHARS = list("CNOSPClBrIF#()[]=123456789+-.cno")
SMILES_CHARS = list("CBNOSPF#-=()[]123456789cno")


def generate_random_smiles(length=20):
    """
    Generates a random (likely invalid) SMILES string of a given length.
    """
    if not SMILES_CHARS:
        raise ValueError("SMILES_CHARS must be defined to generate random SMILES.")
    return "".join(random.choice(SMILES_CHARS) for _ in range(length))


def mutate(smiles_string, mutation_rate):
    """
    Mutates a SMILES string.
    With probability `mutation_rate`, performs one mutation operation:
    1. Character replacement
    2. Character insertion
    3. Character deletion
    4. SMILES branching
    5. SMILES ring formation (cyclization)
    """
    # Characters that, if removed or replaced by 'replace' or 'delete' mutations,
    # should trigger a cleanup attempt.
    critical_chars_for_cleanup = "()[]123456789"

    if random.random() < mutation_rate:
        mutated_list = list(smiles_string)
        needs_cleanup = False  # Flag to indicate if cleanup is needed

        # Handle empty string case for mutation
        if not mutated_list:
            if SMILES_CHARS:
                # Insert a random char if empty; no cleanup needed for this simple case
                return random.choice(SMILES_CHARS)
            return ""  # Cannot mutate if no chars and no SMILES_CHARS

        mutation_type = random.choice(["replace", "insert", "delete", "branch", "ring"])

        if mutation_type == "ring":
            # This mutation inserts a ring.
            # Requires len(mutated_list) >= 1.
            if len(mutated_list) >= 1:
                insert_pos_1 = random.randint(0, len(mutated_list) - 1)
                insert_pos_2 = random.randint(insert_pos_1, len(mutated_list) - 1)

                ring_char_to_use = "1"  # Default
                # Find a digit not currently in the SMILES string
                for i_rc in range(1, 10):
                    if str(i_rc) not in mutated_list:
                        ring_char_to_use = str(i_rc)
                        break

                segment_to_enclose = mutated_list[insert_pos_1:insert_pos_2]
                char_to_append = mutated_list[insert_pos_2]
                remaining_list = mutated_list[insert_pos_2 + 1 :]
                mutated_list = (
                    mutated_list[:insert_pos_1]
                    + [ring_char_to_use]
                    + segment_to_enclose
                    + [ring_char_to_use]
                    + [char_to_append]
                    + remaining_list
                )
            else:
                # This fallback is from the original structure if len < 1 for ring.
                # Given the initial check for empty mutated_list, this path is unlikely
                # unless the initial string was empty and SMILES_CHARS was also empty.
                mutation_type = "insert"

        if mutation_type == "replace" and mutated_list:
            idx_to_replace = random.randrange(len(mutated_list))
            char_being_replaced = mutated_list[idx_to_replace]
            if SMILES_CHARS:  # Ensure SMILES_CHARS is not empty
                mutated_list[idx_to_replace] = random.choice(SMILES_CHARS)
                if char_being_replaced in critical_chars_for_cleanup:
                    needs_cleanup = True
            # If SMILES_CHARS is empty, no replacement occurs.

        if mutation_type == "insert" and SMILES_CHARS:
            # This block follows the structure from the prompt's original mutate function
            _insert_pos = random.randint(0, len(mutated_list))
            _insert_char = random.choice(SMILES_CHARS)
            # Note: SMILES_CHARS in this file uses "()" not "[]()"
            if _insert_char in "()":
                mutation_type = "branch"
            elif _insert_char in "123456789":
                mutation_type = "ring"
            else:
                mutated_list.insert(_insert_pos, _insert_char)

        if mutation_type == "delete":
            if len(mutated_list) > 1:
                del_pos = random.randrange(len(mutated_list))
                char_being_deleted = mutated_list[del_pos]
                del mutated_list[del_pos]
                if char_being_deleted in critical_chars_for_cleanup:
                    needs_cleanup = True
            elif (
                len(mutated_list) == 1 and SMILES_CHARS
            ):  # If only one char, replace it
                char_being_replaced = mutated_list[0]
                mutated_list[0] = random.choice(SMILES_CHARS)
                if char_being_replaced in critical_chars_for_cleanup:
                    needs_cleanup = True

        if mutation_type == "branch" and len(mutated_list) > 0:
            # This mutation wraps a segment and appends a character.
            insert_pos_1 = random.randint(0, len(mutated_list) - 1)
            insert_pos_2 = random.randint(insert_pos_1, len(mutated_list) - 1)

            segment_to_wrap = mutated_list[
                insert_pos_1:insert_pos_2
            ]  # Can be empty if insert_pos_1 == insert_pos_2
            char_to_append = mutated_list[insert_pos_2]
            remaining_list = mutated_list[insert_pos_2 + 1 :]
            mutated_list = (
                mutated_list[:insert_pos_1]
                + ["("]
                + segment_to_wrap
                + [")"]
                + [char_to_append]
                + remaining_list
            )

        final_smiles = "".join(mutated_list)
        if needs_cleanup:
            final_smiles = cleanup_smiles_attempt(final_smiles)
        return final_smiles

    return smiles_string  # No mutation occurred


def cleanup_smiles_attempt(smiles_string: str) -> str:
    """
    Attempts to clean up a SMILES string to improve its chances of being valid.
    This is a heuristic approach and may not always succeed or might significantly
    alter the intended molecule.

    Operations:
    1. Balances parentheses:
       - If open_paren > close_paren, appends necessary ')' at the end.
       - If close_paren > open_paren, removes trailing ')' characters.
    2. Removes empty "()" pairs iteratively.
    3. Ring cleanup: If a ring digit ('1'-'9') appears an odd number of times,
       all occurrences of that specific digit are removed from the string.

    Args:
        smiles_string (str): The SMILES string to clean.

    Returns:
        str: The potentially cleaned SMILES string.
    """
    if not smiles_string:
        return ""

    s_list = list(smiles_string)

    # 1. Balance parentheses
    open_paren_count = s_list.count("(")
    close_paren_count = s_list.count(")")

    if open_paren_count > close_paren_count:
        s_list.extend([")"] * (open_paren_count - close_paren_count))
    elif close_paren_count > open_paren_count:
        diff = close_paren_count - open_paren_count
        while diff > 0 and s_list and s_list[-1] == ")":
            s_list.pop()
            diff -= 1

    open_paren_count = s_list.count("[")
    close_paren_count = s_list.count("]")

    if open_paren_count > close_paren_count:
        s_list.extend(["]"] * (open_paren_count - close_paren_count))
    elif close_paren_count > open_paren_count:
        diff = close_paren_count - open_paren_count
        while diff > 0 and s_list and s_list[-1] == "]":
            s_list.pop()
            diff -= 1
    current_s = "".join(s_list)

    # 2. Remove empty "()" pairs iteratively
    prev_s = None
    while "()" in current_s and current_s != prev_s:
        prev_s = current_s
        current_s = current_s.replace("()", "")

    s_list = list(current_s)
    while "()" in current_s and current_s != prev_s:
        prev_s = current_s
        current_s = current_s.replace("[]", "")

    s_list = list(current_s)

    # 3. Ring cleanup: if a digit '1'-'9' appears an odd number of times, remove all its instances.
    ring_digits = "123456789"
    # Count digits in the current state of s_list
    digit_counts = {digit: s_list.count(digit) for digit in ring_digits}

    digits_to_remove_completely = set()
    for digit, count in digit_counts.items():
        if (
            count > 0 and count % 2 != 0
        ):  # Only consider digits present and with odd counts
            digits_to_remove_completely.add(digit)

    if digits_to_remove_completely:
        s_list = [char for char in s_list if char not in digits_to_remove_completely]

    return "".join(s_list)


def dream_smiles(
    n,
    initial_length_range=(10, 20),
    max_attempts=10000,
    mutation_attempts_per_candidate=5,
):
    """
    Attempts to generate a list of n random but valid SMILES strings by combining
    random generation with a mutation operation.

    For each of the n SMILES to be generated:
    In each attempt (up to max_attempts), a new random SMILES string is generated based on
    `initial_length_range`, and then subjected to one mutation operation
    (mutation rate is 1.0 for these mutations). The mutated string is then cleaned up
    and checked for chemical validity.

    Args:
        n (int): The number of SMILES strings to produce.
        initial_length_range (tuple): A tuple (min_len, max_len) for the length of the
                                      initial random SMILES string generated in each attempt.
        max_attempts (int): Maximum number of attempts to find a valid SMILES string.

        mutation_attempts_per_candidate (int): Number of times to apply mutation to a freshly
                                               generated random SMILES string within one attempt.
    Returns:
        list: A list of n SMILES strings. If a valid SMILES cannot be generated
              for a particular slot within `max_attempts`, `None` will be placed
              in that slot. This function relies on `generate_random_smiles` which
              will raise a ValueError if `SMILES_CHARS` is not defined.
    """
    generated_smiles_list = []
    for _ in range(n):
        found_smiles = None
        for attempt_idx in range(max_attempts):
            min_len, max_len = initial_length_range
            # Ensure valid length for random.randint and meaningful generation
            current_min_len = max(1, min_len)
            current_max_len = max(current_min_len, max_len)

            length = random.randint(current_min_len, current_max_len)

            # generate_random_smiles will raise ValueError if SMILES_CHARS is empty
            current_candidate = generate_random_smiles(length=length)

            # Apply mutation to the newly generated string
            mutated_candidate = current_candidate
            for _ in range(mutation_attempts_per_candidate):
                mutated_candidate = mutate(
                    mutated_candidate, 1.0
                )  # Mutation rate of 1.0 for dreaming

            # Attempt to clean up the mutated SMILES
            cleaned_candidate = cleanup_smiles_attempt(mutated_candidate)

            # Check for chemical validity.
            # Chem.MolFromSmiles returns None for invalid SMILES (including empty strings).
            mol = Chem.MolFromSmiles(cleaned_candidate)
            if cleaned_candidate and mol is not None:
                try:
                    found_smiles = Chem.MolToSmiles(mol, canonical=True)
                except Exception:  # In case canonicalization fails for some edge reason
                    found_smiles = cleaned_candidate  # Fallback to the cleaned, non-canonical version
                break  # Found a valid SMILES for this slot
        generated_smiles_list.append(
            found_smiles
        )  # Add found SMILES or None if not found
    return generated_smiles_list


def crossover(parent1, parent2, crossover_rate):
    """
    Performs crossover between two parent SMILES strings using a random cut point for each.
    """
    if random.random() < crossover_rate:
        if (
            not parent1 or not parent2
        ):  # Cannot crossover with an empty string effectively
            return parent1, parent2

        # Determine cut points, allowing for full string if length is 1
        p1_cut = random.randint(0, len(parent1)) if len(parent1) > 0 else 0
        p2_cut = random.randint(0, len(parent2)) if len(parent2) > 0 else 0

        child1 = parent1[:p1_cut] + parent2[p2_cut:]
        child2 = parent2[:p2_cut] + parent1[p1_cut:]
        return child1, child2
    return parent1, parent2


def tournament_selection(population_with_fitness, tournament_size):
    """
    Selects an individual using tournament selection.
    Assumes fitness is the first element of tuples in population_with_fitness,
    and higher fitness is better.
    """
    if not population_with_fitness:
        return None

    actual_tournament_size = min(tournament_size, len(population_with_fitness))
    if (
        actual_tournament_size == 0
    ):  # Should not happen if population_with_fitness is not empty
        return None

    tournament = random.sample(population_with_fitness, actual_tournament_size)
    tournament.sort(key=lambda x: x[0], reverse=True)  # Sort by fitness, descending
    return tournament[0][1]  # Return the SMILES string of the winner


def simple_genetic_algorithm(
    initial_population,
    fitness_function,
    generations,
    population_size,
    crossover_rate,
    mutation_rate,
    tournament_size,
    elitism_count=1,
    random_seed=42,
):
    """
    A simple genetic algorithm for SMILES string optimization.

    Args:
        initial_population (list): A list of initial SMILES strings.
        fitness_function (callable): A function that takes a SMILES string
                                     and returns a fitness score (higher is better).
                                     It should handle invalid SMILES (e.g., return a very low score or None).
        generations (int): The number of generations to run.
        population_size (int): The target size of the population.
        crossover_rate (float): The probability of crossover for a pair of parents.
        mutation_rate (float): The probability an individual undergoes mutation.
        tournament_size (int): The number of individuals in a selection tournament.
        elitism_count (int): Number of best individuals to carry over to the next generation.
        random_seed (int): Seed for the random number generator for reproducibility.

    Returns:
        dict: A dictionary containing the results of the GA run:
              - "best_smiles" (str): The best SMILES string found.
              - "best_fitness" (float): The fitness score of the best SMILES string.
              - "all_fitness_scores" (list): A list of lists, where each inner list
                                             contains the fitness scores of all individuals
                                             in that generation.
              - "best_fitness_history" (list): A list of the best fitness score in each generation.
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Population initialization
    population = list(initial_population) if initial_population else []

    # Ensure population is of population_size
    if len(population) < population_size:
        num_to_generate = population_size - len(population)

        if not SMILES_CHARS and num_to_generate > 0 and not population:
            raise ValueError(
                "SMILES_CHARS must be defined to generate random initial population if initial_population is empty."
            )

        # Attempt to generate individuals using dream_smiles if SMILES_CHARS are available
        if SMILES_CHARS and num_to_generate > 0:
            print(
                f"Initial population too small ({len(population)}/{population_size}). Trying to generate {num_to_generate} individuals using dream_smiles..."
            )
            # Using default parameters for dream_smiles as defined in its signature:
            # initial_length_range=(10, 20), max_attempts=10000, mutation_attempts_per_candidate=5
            generated_by_dream = dream_smiles(n=num_to_generate)

            valid_dreamed_smiles = [s for s in generated_by_dream if s is not None]
            population.extend(valid_dreamed_smiles)
            print(
                f"Added {len(valid_dreamed_smiles)} individuals from dream_smiles. Population size: {len(population)}/{population_size}"
            )
            num_to_generate = population_size - len(
                population
            )  # Update remaining needed

        # Fill remaining spots if dream_smiles didn't produce enough,
        # or if SMILES_CHARS was not available for dream_smiles initially (less likely due to above check).
        if num_to_generate > 0:
            print(
                f"Still need to generate {num_to_generate} individuals using fallback methods..."
            )
            for _ in range(num_to_generate):
                if (
                    population
                ):  # Mutate existing if possible (either from initial_pop or dream_smiles)
                    population.append(mutate(random.choice(population), 1.0))
                elif (
                    SMILES_CHARS
                ):  # Fallback to simpler generation if population is still empty
                    # (e.g. initial_pop was empty and dream_smiles yielded nothing)
                    population.append(generate_random_smiles(random.randint(10, 30)))
                else:
                    # This case implies SMILES_CHARS is not defined, and population is empty.
                    # The initial ValueError should have caught this if initial_population was also empty.
                    # If initial_population was not empty, but SMILES_CHARS is not defined,
                    # the 'if population:' branch above would be taken.
                    break  # Avoid issues if somehow in an unhandled state
    elif len(population) > population_size:
        population = random.sample(population, population_size)  # Sample if too large

    current_best_smiles = None
    current_best_fitness = -float("inf")

    all_generation_fitness_scores = []
    best_fitness_per_generation = []

    for gen in range(generations):
        pop_with_fitness = []
        for smiles in population:
            fitness = fitness_function(smiles)
            if fitness is not None:  # Allow fitness function to reject individuals
                pop_with_fitness.append((fitness, smiles))

        if not pop_with_fitness:
            print(
                f"Generation {gen+1}: No individuals with valid fitness scores. Stopping."
            )
            break

        pop_with_fitness.sort(key=lambda x: x[0], reverse=True)

        # Store fitness data for the current generation
        all_generation_fitness_scores.append([item[0] for item in pop_with_fitness])
        best_fitness_per_generation.append(pop_with_fitness[0][0])

        if pop_with_fitness[0][0] > current_best_fitness:
            current_best_fitness = pop_with_fitness[0][0]
            current_best_smiles = pop_with_fitness[0][1]
            print(
                f"Generation {gen+1}: New best -> Fitness: {current_best_fitness:.4f}, SMILES: {current_best_smiles}"
            )
        else:
            # Log current generation's best if no improvement overall
            print(
                f"Generation {gen+1}: Best this gen -> Fitness: {pop_with_fitness[0][0]:.4f}, SMILES: {pop_with_fitness[0][1]}"
            )

        new_population = []

        # Elitism
        if elitism_count > 0:
            elites = [
                ind[1]
                for ind in pop_with_fitness[: min(elitism_count, len(pop_with_fitness))]
            ]
            new_population.extend(elites)

        # Generate new individuals
        while len(new_population) < population_size:
            parent1_smiles = tournament_selection(pop_with_fitness, tournament_size)
            parent2_smiles = tournament_selection(pop_with_fitness, tournament_size)

            if (
                parent1_smiles is None or parent2_smiles is None
            ):  # Should only happen if pop_with_fitness is empty
                print(
                    f"Warning: Parent selection failed in generation {gen+1}. Populating with random individuals."
                )
                # Fallback: if selection fails (e.g. pop_with_fitness became unexpectedly small)
                # add random or mutated individuals. For simplicity, break if no parents.
                if not pop_with_fitness:
                    break
                parent1_smiles = random.choice(pop_with_fitness)[1]
                parent2_smiles = random.choice(pop_with_fitness)[1]

            child1_smiles, child2_smiles = crossover(
                parent1_smiles, parent2_smiles, crossover_rate
            )

            new_population.append(mutate(child1_smiles, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutate(child2_smiles, mutation_rate))

        population = new_population[:population_size]  # Ensure population size

        if not population:
            print(f"Generation {gen+1}: Population became empty. Stopping.")
            break

    return {
        "best_smiles": current_best_smiles,
        "best_fitness": current_best_fitness,
        "all_fitness_scores": all_generation_fitness_scores,
        "best_fitness_history": best_fitness_per_generation,
    }


# Plotting functions
def plot_ga_results(results):
    if not results:
        print("No results to plot.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")  # Apply a nicer style
    # Visualize the best molecule found
    print("\nVisualizing the best molecule...")
    if results["best_smiles"]:
        mol = Chem.MolFromSmiles(results["best_smiles"])
        if mol:
            if _HAS_CAIRO:
                img = Draw.MolToImage(mol, size=(300, 300))
                plt.figure(figsize=(4, 4))
                plt.imshow(img)
                plt.axis("off")
                plt.title(
                    f"Best Molecule Structure\nSMILES: {results['best_smiles']}\nFitness: {results['best_fitness']:.2f}"
                )
                plt.show()
            else:
                img = _mol_grid_image([mol], mols_per_row=1, sub_img_size=(300, 300))
                print(
                    f"Best Molecule Structure\nSMILES: {results['best_smiles']}\nFitness: {results['best_fitness']:.2f}"
                )
                _display_mol_grid(img)
        else:
            print(
                f"Could not generate RDKit molecule object for SMILES: {results['best_smiles']}"
            )
    else:
        print("No best SMILES found to visualize.")
    if results["best_fitness_history"]:
        plt.figure(figsize=(12, 6))
        generations_ran = len(results["best_fitness_history"])

        # Plot 1: Best and Average Fitness per Generation
        plt.subplot(1, 2, 1)
        plt.plot(
            results["best_fitness_history"],
            marker="o",
            linestyle="-",
            color="dodgerblue",
            label="Best Fitness",
        )

        if results["all_fitness_scores"]:
            avg_fitness_per_generation = [
                sum(scores) / len(scores)
                for scores in results["all_fitness_scores"]
                if scores
            ]
            if avg_fitness_per_generation:
                plt.plot(
                    avg_fitness_per_generation,
                    marker="x",
                    linestyle="--",
                    color="green",
                    label="Average Fitness",
                )

        plt.title("Fitness Progression Over Generations")
        plt.xlabel(f"Generation (0-{generations_ran-1})")
        plt.ylabel("Fitness Score")
        plt.legend()
        plt.grid(True)

        # Plot 2: Fitness Distribution per Generation (Box Plot)
        if results["all_fitness_scores"]:
            plt.subplot(1, 2, 2)
            valid_generation_scores = [
                gen_scores for gen_scores in results["all_fitness_scores"] if gen_scores
            ]
            if valid_generation_scores:
                bp = plt.boxplot(
                    valid_generation_scores,
                    patch_artist=True,
                    medianprops={"color": "black"},
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor("lightblue")
                plt.title("Fitness Distribution per Generation")
                plt.xlabel(f"Generation (0-{generations_ran-1})")
                plt.ylabel("Fitness Score")
                step = max(1, generations_ran // 10 if generations_ran > 10 else 1)
                plt.xticks(
                    ticks=range(1, generations_ran + 1, step),
                    labels=[str(i - 1) for i in range(1, generations_ran + 1, step)],
                )
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No fitness distributions to plot.",
                    ha="center",
                    va="center",
                )

        plt.tight_layout()
        plt.show()
    else:
        print("No fitness history to plot.")


def visualize_smiles_list(
    smiles_list, mols_per_row=4, sub_img_size=(200, 200), legends=None
):
    """
    Visualizes a list of SMILES strings as a grid of 2D molecular images
    in a Jupyter Notebook.

    Args:
        smiles_list (list): A list of SMILES strings.
        mols_per_row (int): Number of molecules to display per row in the grid.
        sub_img_size (tuple): Tuple (width, height) for each individual molecule image.
        legends (list, optional): A list of strings to use as legends for each molecule.
                                  Should be the same length as smiles_list. Defaults to None.
    """
    mols = []
    valid_legends = []
    for i, smiles in enumerate(smiles_list):
        if not smiles:  # Handle empty or None SMILES strings
            print(f"SMILES string at index {i} is empty or None. Skipping.")
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mols.append(mol)
            if legends and i < len(legends):
                valid_legends.append(str(legends[i]))
            else:
                # Use SMILES string as legend if no specific legend is provided or if out of bounds
                valid_legends.append(smiles)
        else:
            print(f"Could not parse SMILES: {smiles}. Skipping.")

    if not mols:
        print("No valid molecules to display.")
        return

    # If legends were not provided or all were invalid, MolsToGridImage might error
    # or produce no legends. Ensure valid_legends matches mols length if used.
    if len(valid_legends) != len(mols):
        current_legends = [
            m.GetProp("_Name") if m.HasProp("_Name") else Chem.MolToSmiles(m)
            for m in mols
        ]
    else:
        current_legends = valid_legends

    img = _mol_grid_image(
        mols, mols_per_row, sub_img_size, legends=current_legends
    )
    if img:
        _display_mol_grid(img)
    else:
        print("Failed to generate image grid.")


# Some example fitness functions
def rdkit_logp_fitness(smiles):
    """
    Calculates the LogP for a given SMILES string using RDKit.
    Returns a very low score for invalid SMILES.
    """
    if not smiles:
        return -1000.0  # Penalize empty SMILES

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1000.0  # Invalid SMILES, return a very low fitness
    try:
        logp = Crippen.MolLogP(mol)
        return logp
    except Exception:  # Catch any other RDKit errors during calculation
        return -1000.0
