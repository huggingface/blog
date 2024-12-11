#!/bin/bash
echo "🚀 You've entered the Infinite Orbit Hole!"
COUNTER=0

while true; do
    ((COUNTER++))
    echo "Round $COUNTER of The Orbit Hole... Still looping. Are you sure you want to continue? (yes/no)"
    read -r user_choice
    if [ "$user_choice" == "no" ]; then
        echo "Starla Moonshadow whispers: 'Wise choice. Exit and reflect on your journey.'"
        break
    else
        echo "Jaxo the Jester cackles: 'Round and round we go!'"
    fi
done

