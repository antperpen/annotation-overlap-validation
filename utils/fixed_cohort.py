import numpy as np
import matplotlib.pyplot as plt

# Indices of the tuples stored per image and overlap in imagesValue
IDX_OVERLAP = 0             # imod                   -> overlap percentage
IDX_NUM_PATCHES = 1         # validationObj.patch_num -> evaluable patches
IDX_CORRECT_PATCHES = 2     # validationObj.A         -> correctly classified patches

# Colors of the fixed cohort figure
COHORT_MEDIAN_COLOR = '#2a78d6'
COHORT_POOLED_COLOR = '#eb6834'
ALL_IMAGES_POOLED_COLOR = '#52514e'

def get_fixed_cohort(images_value):
    """
    Groups the results of imagesValue by image and overlap and selects the fixed cohort: the
    images with evaluable patches at every overlap. Returns the sorted overlaps, the results of
    all the images and those of the cohort ({image: {overlap: (patches, correct)}})
    """
    per_image = {image: {entry[IDX_OVERLAP]: (entry[IDX_NUM_PATCHES], entry[IDX_CORRECT_PATCHES]) for entry in entries}
                 for image, entries in images_value.items()}
    overlaps = sorted({overlap for values in per_image.values() for overlap in values})
    cohort = {image: values for image, values in per_image.items()
              if all(values.get(overlap, (0, 0))[0] > 0 for overlap in overlaps)}
    return overlaps, per_image, cohort

def pooled_accuracy(per_image, overlap):
    """Pooled accuracy at an overlap: correctly classified patches over evaluable patches of all the images"""
    counts = [values[overlap] for values in per_image.values() if overlap in values]
    num_patches = sum(count[0] for count in counts)
    return sum(count[1] for count in counts) / num_patches if num_patches else float('nan')

def plot_fixed_cohort_accuracy(images_value, title, output_path):
    overlaps, per_image, cohort = get_fixed_cohort(images_value)
    if not cohort:
        print("No image has evaluable patches at every overlap: {} is not generated".format(output_path))
        return
    accuracy = np.array([[values[overlap][1] / values[overlap][0] for overlap in overlaps] for values in cohort.values()])
    q1, q3 = np.percentile(accuracy, [25, 75], axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(overlaps, q1, q3, color=COHORT_MEDIAN_COLOR, alpha=0.15, linewidth=0, zorder=2,
                    label='Interquartile range (cohort)')
    ax.plot(overlaps, np.median(accuracy, axis=0), color=COHORT_MEDIAN_COLOR, linewidth=2.5, marker='o',
            markersize=7, zorder=4, label='Median per image (cohort)')
    ax.plot(overlaps, [pooled_accuracy(cohort, overlap) for overlap in overlaps], color=COHORT_POOLED_COLOR,
            linewidth=2, marker='s', markersize=7, zorder=4, label='Pooled accuracy (cohort)')
    # Zorder above "Pooled accuracy (cohort)" and a thicker, more marked dash: in small datasets
    # (e.g. Valme, 5 images) the cohort excludes few images with few patches and this line almost
    # matches the orange one, being hidden below it if it is not highlighted
    ax.plot(overlaps, [pooled_accuracy(per_image, overlap) for overlap in overlaps], color=ALL_IMAGES_POOLED_COLOR,
            linewidth=2.5, linestyle=(0, (6, 3)), zorder=6, label='Pooled accuracy (all images)')

    ax.set_xticks(overlaps)
    ax.set_xlabel('Percentage of overlap', fontsize=12, labelpad=8)
    ax.set_ylabel('Percentage of successful patches', fontsize=12, labelpad=8)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.grid(True, linestyle='--', color='lightgray', alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    # Same font weight as the titles of the other figures of run_validation.py (normal, not bold)
    ax.set_title('{}\nFixed cohort: {} of {} images with patches at every overlap'.format(title, len(cohort), len(per_image)),
                 fontsize=13, fontweight='normal', pad=14)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure saved in: {}".format(output_path))
