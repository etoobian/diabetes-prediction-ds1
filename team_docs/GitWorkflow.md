# Team Git & GitHub Workflow

**Goal:** We use branches + pull requests (PRs) so `main` stays stable/runnable, and changes get reviewed before merging.

- Everything is merged through GitHub Pull Requests.
- Branches should represent **one small, complete task**.
- Do not push directly to `main`.

---

## 1. Clone the Team Repository (One Time)

```bash
cd ~/<DIRECTORY>
mkdir class_project
cd class_project
git clone <TEAM_REPO_URL>
cd <REPO_NAME>
```

---

## 2. Sync main Before You Start (Every Work Session)

```bash
git checkout main
git pull
```

This ensures you branch off the most up-to-date main.

## 3. Create Your Personal Work Branch (One Task Per Branch)

```bash
git checkout -b your-name/task-description
git push -u origin your-name/task-description
```

**NOTES:** 
  
  - `checkout -b` creates new branch and *switches* to it
  - `push -u origin` pushes your branch to GitHub and sets upstream for future `git push` commands
  - Replace `your-name/task-description` with something short and meaningful (i.e. `name/feature` or `name/fix`)
  

Example:

```bash
git checkout -b esther/add-eda-section
git push -u origin esther/add-eda-section
```

---

## 4. Start Working

Make changes to files normally.

---

## 5. Save Your Work to Git (Commit Often)

```bash
git add <SPECIFIC FILES TO ADD>
git commit -m "Short description of changes"
git push
```

**Tip:** Prefer adding specific files rather than `git add .` until you’re comfortable.

---

## 6. If main Changes While You’re Working (Occasionally)

If your PR shows conflicts, or your branch is behind `main`, update your branch like this:

```bash
git checkout your-name/task-description
git pull origin main
# resolve conflicts if needed
git push
```

(For now, we’ll keep it simple and avoid rebasing.)

---

## 7. Open a Pull Request (PR)

1. Go to GitHub, then to your repo
2. Click **Pull Requests**
3. Click **New Pull Request**
4. Base branch: `main`
5. Compare branch: your personal branch
6. Click **Create Pull Request**
7. Title your PR clearly (what changed + why)

---

## 8. Request Reviews

On the PR page:
  - Click **Reviewers**
  - Add the other teammate (required)

---

## 9. Review Process

Reviewers will choose:
  - **Approve**
  - **Comment**
  - **Request Changes**

If changes are requested:

```bash
git add .
git commit -m "Address review feedback"
git push
```

**Notes:**
  - Keep Code reviews constructive.
  - All review comments must be addressed before merging. 
  - If commits are added after approval, a **re-review must be requested**.

---

## 10. Merge Only When Approved

After approval:

  - Click **Merge Pull Request**
  - Click **Confirm Merge**
  - Delete branch after merging (GitHub will suggest this)
  - Delete local branch 
      - Move to `main` branch: `git checkout main`
      - Delete local branch: `git branch -d <branch name>`

```bash
git checkout main
git pull
git branch -d <branch-name>
```

---

## 11. Update Local Main After Merge

Make sure on main: `git branch`

If not, move to main: `git checkout main`

Update Local main: `git pull`

---

## 12. Start Next Task

Repeat from Step 2 for every new task.