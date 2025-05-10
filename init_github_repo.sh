#!/bin/bash
# Initialize a new GitHub repository and push the initial commit

# Set your GitHub username
read -p "Enter your GitHub username: " github_username

# Set repository name
repo_name="aimThat"

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
else
    echo "Git repository already initialized."
fi

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: aimThat"

# Create a new repository on GitHub (requires GitHub CLI)
# Uncomment if you have GitHub CLI installed
# echo "Creating repository on GitHub..."
# gh repo create $repo_name --public --description "Machine learning component of the NoScope9000 sniper shot prediction system"

# Add GitHub remote
echo "Adding GitHub remote..."
git remote add origin "https://github.com/$github_username/$repo_name.git"

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main

echo "Done! Your repository is now available at: https://github.com/$github_username/$repo_name"
echo "If you see any errors, you may need to create the repository on GitHub first."
echo "You can do this by visiting: https://github.com/new"
