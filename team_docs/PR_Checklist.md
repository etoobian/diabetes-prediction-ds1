# Pull Request Checklist

Before requesting a review, confirm each item:

- [ ] My code builds/runs without errors
- [ ] No debugging print statements left in code
- [ ] I tested my code on at least one real input /scenario
- [ ] I pulled latest `main` into my branch if needed and resolved merge conflicts locally
- [ ] My branch name reflects the task (e.g., `name/task-description`)
- [ ] My PR title describes the change clearly
- [ ] My PR is small enough to review (ideally under ~200 lines changed, if possible)
- [ ] I requested **the other teammate** as reviewer
- [ ] I updated documentation if needed / README / comments if needed

**After approval:**
- [ ] I merged using the GitHub PR (not terminal push to `main`)
- [ ] I deleted the remote branch (button on GitHub)
- [ ] I deleted the remote branch (button on GitHub)

```bash
git checkout main
git pull
git branch -d <branch-name>
```

- [ ] I updated my local `main`:

```bash
git checkout main
git pull
```