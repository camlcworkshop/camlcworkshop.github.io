import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut, KFold
from scipy import stats 
import numpy as np
import seaborn as sns

def calculate_r2(y_true, y_pred):
    res = stats.linregress(y_true, y_pred)
    r2 = res.rvalue**2

    return r2

def r2_val(y_test,y_pred_test,y_train):
    """Calculates the external R2 pred as described:
    https://pdfs.semanticscholar.org/4eb2/5ff5a87f2fd6789c5b9954eddddfd1c59dab.pdf"""
    y_resid = y_pred_test - y_test
    SS_resid = np.sum(y_resid**2)
    y_var = y_test - np.mean(y_train)
    SS_total = np.sum(y_var**2)
    r2_validation = 1-SS_resid/SS_total
    return(r2_validation)

def plot_model(y_train, y_test, y_train_pred, y_test_pred):
    plt.figure(figsize=(8,8))

    # Plot the training and test predictions
    plt.scatter(y_train, y_train_pred, label='Training Set', s=120, edgecolor='black')
    plt.scatter(y_test, y_test_pred, label='Test Set', s=120, edgecolor='black', marker='^')

    # Plot the ideal line
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='black', linewidth=2, label='Ideal Line')

    # Get model statistics
    r2_train = calculate_r2(y_train, y_train_pred)
    r2_test = r2_val(y_test, y_test_pred, y_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    stats_text = (
        f"Training R²: {r2_train:.3f}\n"
        f"Test R²: {r2_test:.3f}\n"
        f"Training MAE: {mae_train:.3f}\n"
        f"Test MAE: {mae_test:.3f}\n"
        f"Training MSE: {mse_train:.3f}\n"
        f"Test MSE: {mse_test:.3f}\n"
        f"Training RMSE: {rmse_train:.3f}\n"
        f"Test RMSE: {rmse_test:.3f}"
    )

    # Plot the MAE error lines
    plt.plot([min(y_train), max(y_train)], [min(y_train) + mae_train, max(y_train) + mae_train], color='gray', linestyle='dashed')
    plt.plot([min(y_train), max(y_train)], [min(y_train) - mae_train, max(y_train) - mae_train], color='gray', linestyle='dashed')

    # Set plot labels
    plt.xlabel('True Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.grid(True)
    plt.legend()

    plt.show()

    print("Model Statistics:")
    print(stats_text)


def get_error_plot(y_train, y_test, y_train_pred, y_test_pred, xlim=(-0.5, 1.5)):
    # Get model errors

    train_errors = []
    for prediction, true in zip(y_train_pred, y_train):
        error = abs(prediction - true)
        train_errors.append(error)

    validation_errors = []
    for prediction, true in zip(y_test_pred, y_test):
        error = abs(prediction - true)
        validation_errors.append(error)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex = True)

    sns.kdeplot(train_errors, ax=axs[0], label="Training set", fill=True)
    sns.kdeplot(validation_errors, ax=axs[0], label="Test set", fill=True)

    axs[1].scatter(train_errors, y_train, label="Training set", s=100, edgecolor='black')
    axs[1].scatter(validation_errors, y_test, label="Test set", s=100, marker='s', edgecolor='black')
    axs[1].axvline(x=mae_train, linestyle=':', linewidth=3)
    axs[1].axvline(x=mae_test, linestyle=':', linewidth=3)

    axs[1].legend()
    axs[1].grid()   

    axs[0].set_ylabel("Density")

    axs[0].axvline(x=mae_train, linestyle=':', linewidth=3, label='Training MAE')
    axs[0].axvline(x=mae_test, linestyle=':', linewidth=3, label='Test MAE')

    # Custom legend
    custom_lines = [
        plt.Line2D([0], [0], linestyle=':', lw=3, label=f'Training MAE\n({mae_train:.2f} kcal mol$^{{-1}}$)'),
        plt.Line2D([0], [0], linestyle=':', lw=3, label='Test MAE\n({:.2f} kcal mol$^{{-1}}$)'.format(mae_test))
    ]

    axs[0].legend(handles=custom_lines, loc='upper right')

    axs[1].set_xlabel("Absolute error \n (|predicted $\Delta \Delta G ^{\u2021}$ $-$ measured $\Delta \Delta G ^{\u2021}$|) / kcal mol$^{-1}$")
    axs[1].set_ylabel("Measured $\Delta \Delta G ^{\u2021}$")

    plt.xlim(xlim)


def run_loocv(dt, X_train, y_train, y_train_pred):
    plt.figure(figsize=(8,8))
    plt.scatter(y_train, y_train_pred, color="k", label="Training Set", s=100)

    # start the leave one out analysis
    loo_mae_scores = []
    loo_rmse_scores = []

    ytests = []
    ypreds = []

    loo = LeaveOneOut()
    for train,test in loo.split(X_train):
        X_train_loo, X_test_loo = X_train[train], X_train[test]
        y_train_loo, y_test_loo = y_train[train], y_train[test]
        dt.fit(X_train_loo, y_train_loo)
        y_pred_loo = dt.predict(X_test_loo)
        plt.scatter(y_test_loo, y_pred_loo, facecolors='none', edgecolors='k', s=100)

        ytests += list(y_test_loo)
        ypreds += list(y_pred_loo)

        mae = mean_absolute_error(y_test_loo, y_pred_loo)
        rmse = root_mean_squared_error(y_test_loo, y_pred_loo)
        loo_mae_scores.append(mae)
        loo_rmse_scores.append(rmse)
        # print(f"R2 (kfold {colors.index(i)}): {r2}")

    r2 = calculate_r2(ytests, ypreds)

    print(f"Leave one out R2: {r2:.2f}")
    print(f"Leave one out MAE: {np.mean(loo_mae_scores):.2f}")
    print(f"Leave one out RMSE: {np.mean(loo_rmse_scores):.2f}")

    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='black')
    plt.xlabel("True values", fontsize=14)
    plt.ylabel("Predicted values", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


def run_foldcv(dt, X_train, y_train, y_train_pred, n_splits=5):
    # First, run the k-fold cross validation
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Get the default color cycle
    matplotlibcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Extract the first five colors, one for each fold
    colors = matplotlibcolors[:n_splits]

    # Plot the k-fold resulr
    fig, axs = plt.subplots(1, n_splits, figsize=(20, 7), sharey=True)

    r2s = []
    maes = []
    rmses = []

    for i, (train_index, test_index) in zip(colors,k_fold.split(X_train)):
        X_train_k, X_test_k = X_train[train_index], X_train[test_index]
        y_train_k, y_test_k = y_train[train_index], y_train[test_index]
        dt.fit(X_train_k, y_train_k)
        y_pred_k = dt.predict(X_test_k)
        axs[colors.index(i)].scatter(y_train, y_train_pred, facecolors=i, edgecolors=i, s=100, label='Train')
        axs[colors.index(i)].scatter(y_test_k, y_pred_k, facecolors='none', edgecolors=i, s=100, label='k-fold')
        axs[colors.index(i)].set_xlabel("True values", fontsize=14)
        axs[colors.index(i)].set_ylabel("Predicted values", fontsize=14)
        axs[colors.index(i)].plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='black')
        axs[colors.index(i)].set_title(f'k-fold {colors.index(i)}')
        axs[colors.index(i)].legend()
        r2 = calculate_r2(y_test_k, y_pred_k)
        # r2 = r2_val(y_test_k, y_pred_k, y_train_k)
        r2s.append(r2)
        mae = mean_absolute_error(y_test_k, y_pred_k)
        maes.append(mae)
        rmse = root_mean_squared_error(y_test_k, y_pred_k)
        rmses.append(rmse)
        axs[colors.index(i)].text(0.95, 0.05, f'k-fold R2: {r2:.2f}\nk-fold MAE: {mae:.2f}\nk-fold RMSE: {rmse:.2f}', transform=axs[colors.index(i)].transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    fig.suptitle(f'k-Fold Cross Validation Results: Average R2 = {np.array(r2s).mean():.2f}; Average MAE = {np.array(maes).mean():.2f}; Average RMSE = {np.array(rmses).mean():.2f}', fontsize=16)
    plt.show()